import re
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Tuple, Any, Union, Optional, Type, Dict, Literal, Generic, TypeVar, Iterable

from langchain import FewShotPromptTemplate, BasePromptTemplate
from langchain import PromptTemplate
from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain.callbacks.base import Callbacks
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLLM
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.schema import AgentAction, AgentFinish, Document
from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel


class BaseNpcSheet(BaseModel, ABC):
    name: str

    @property
    @abstractmethod
    def summary(self) -> str:
        pass


class KnowledgeDocTypes(Enum):
    BLANK = 1
    SCENE = 2

    @classmethod
    def blank(cls) -> Document:
        return Document(page_content="no information", metadata={"type": cls.BLANK})

    @classmethod
    def scene(cls, summary: str) -> Document:
        return Document(page_content=summary, metadata={"type": KnowledgeDocTypes.SCENE})


class KnowledgeSummarizer(BaseCombineDocumentsChain):
    llm: BaseLLM
    template = PromptTemplate.from_template(textwrap.dedent("""
        I have taken the following mental notes:
        {notes}

        Please write a short paragraph, from my own point of view.
        It should summarize concisely the notes mentioned above.
    """).strip())

    async def acombine_docs(self, docs: List[Document], **kwargs: Any) -> Tuple[str, dict]:
        return self.combine_docs(docs, **kwargs)

    def combine_docs(self, docs: List[Document], **kwargs: Any) -> Tuple[str, dict]:
        if not docs:
            return "no information was found", {}
        notes = [f"NOTE #{i}:\n{d.page_content.strip()}" for i, d in enumerate(docs)]
        prompt = self.template.format(notes="\n\n".join(notes))
        return self.llm(prompt).strip(), {}


@dataclass
class NpcKnowledge:
    vectorstore: VectorStore
    summarizer: BaseCombineDocumentsChain
    max_retrieval: int
    min_relevance: float

    def query_docs(self, query) -> List[Tuple[Document, float]]:
        return self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=self.max_retrieval,
            score_threshold=self.min_relevance,
        )

    def query_summarized(self, query) -> str:
        docs = self.query_docs(query)
        return self.summarizer.run([d[0] for d in docs])


@dataclass
class Npc:
    sheet: BaseNpcSheet
    knowledge_base: NpcKnowledge
    situation: str = ""


class RolePlayAction(BaseModel, ABC):
    actor: Union[str, Literal["self"]]

    def as_own(self):
        return self.with_actor("self")

    def with_actor(self, actor: str):
        kwargs = self.dict()
        kwargs["actor"] = actor
        return self.construct(**kwargs)

    @classmethod
    def own_action(cls, **kwargs):
        return cls(actor="self", **kwargs)

    @property
    def is_own_action(self) -> bool:
        return self.actor == "self"

    @property
    @abstractmethod
    def action_description(self) -> str:
        pass


class NoAction(RolePlayAction):

    @property
    def action_description(self) -> str:
        if self.is_own_action:
            return "I do nothing"
        else:
            return f"{self.actor} does nothing"


class SpeechAction(RolePlayAction):
    text: str

    @property
    def action_description(self) -> str:
        if self.is_own_action:
            return f"I said: {self.text}"
        else:
            return f"{self.actor} said: {self.text}"


class FreeAction(RolePlayAction):
    text: str

    @property
    def action_description(self) -> str:
        if self.is_own_action:
            return f"I did the following: {self.text}"
        else:
            return f"{self.actor} did the following: {self.text}"


A = TypeVar("A", bound=RolePlayAction)


class RolePlayTool(Tool, Generic[A], ABC):

    @classmethod
    def default_func(cls, x: str) -> str:
        return x

    @classmethod
    @abstractmethod
    def create(cls) -> "RolePlayTool[A]":
        pass

    @abstractmethod
    def role_play_action(self, agent_action: AgentAction) -> A:
        pass


class DoNothingTool(RolePlayTool):

    @classmethod
    def create(cls) -> RolePlayTool[NoAction]:
        return cls(
            name="Wait",
            description="end the loop without doing or saying anything",
            func=cls.default_func,
        )

    def role_play_action(self, agent_action: AgentAction) -> NoAction:
        return NoAction.own_action()


class SpeechTool(RolePlayTool):

    @classmethod
    def create(cls) -> RolePlayTool[NoAction]:
        return cls(
            name="Say",
            description="end the loop by saying something",
            func=cls.default_func,
        )

    def role_play_action(self, agent_action: AgentAction) -> SpeechAction:
        return SpeechAction.own_action(text=agent_action.tool_input)


class FreeActionTool(RolePlayTool):

    @classmethod
    def create(cls) -> RolePlayTool[NoAction]:
        return cls(
            name="Do",
            description="end the loop by doing something. describe what you do",
            func=cls.default_func,
        )

    def role_play_action(self, agent_action: AgentAction) -> FreeAction:
        return FreeAction.own_action(text=agent_action.tool_input)


@dataclass
class ReActExample:
    content: str

    def __post_init__(self):
        self.content = textwrap.dedent(self.content).strip()


class ReActUtils:

    ACTION_EXP = re.compile(r"^\[(?P<tool>\w+)] (?P<tool_input>.*)$")

    @classmethod
    def parse_react_history(cls, intermediate_steps: List[Tuple[AgentAction, str]]) -> str:
        return "\n".join([
            f"Act: [{s[0].tool}] {s[0].tool_input}\nObs: {s[1].strip()}"
            for s in intermediate_steps
        ])

    @classmethod
    def extract_action(cls, answer: str) -> AgentAction:
        clean_answer = answer.split("\n")[0].strip()
        try:
            match_groups = cls.ACTION_EXP.match(clean_answer).groupdict()
            return AgentAction(**match_groups, log="\n" + clean_answer + "\n")
        except Exception as exc:
            raise RuntimeError(f"Cannot parse agent action: {clean_answer}") from exc

    @classmethod
    def create_similarity_examples_selector(
        cls,
        examples: List[ReActExample],
        embeddings: Embeddings,
        vectorstore_cls: Type[VectorStore],
        num_examples: int,
    ) -> SemanticSimilarityExampleSelector:
        return SemanticSimilarityExampleSelector.from_examples(
            examples=[asdict(e) for e in examples],
            vectorstore_cls=vectorstore_cls,
            embeddings=embeddings,
            k=num_examples,
        )


class NpcAgent(BaseSingleActionAgent):
    npc: Npc
    tools: List[Tool]
    llm: BaseLLM
    example_selector: BaseExampleSelector
    tool_lookup: Optional[Dict[str, Tool]] = None
    template: Optional[BasePromptTemplate] = None

    class Config:
        arbitrary_types_allowed: bool = True

    def __init__(self, npc: Npc, role_play_tools: List[RolePlayTool], **kwargs):
        assert role_play_tools, "At least one role play tool must be provided"
        super().__init__(
            npc=npc,
            tools=self.default_tools(npc) + role_play_tools,
            **kwargs
        )
        self.template = self.create_template()
        self.tool_lookup = {t.name: t for t in self.tools}

    @classmethod
    def default_tools(cls, npc: Npc) -> List[Tool]:
        return [
            Tool(
                name="Think",
                description="make an observation to yourself",
                func=lambda x: "OK",
            ),
            Tool(
                name="Recall",
                description="recall something from your memory or personality",
                func=lambda x: npc.knowledge_base.query_summarized(x),
            ),
        ]

    def create_template(self) -> BasePromptTemplate:
        actions = '\n\t\t'.join([f"{t.name}: {t.description}" for t in self.tools])
        return FewShotPromptTemplate(
            prefix=textwrap.dedent(f"""
                {{character_summary}}

                Answer with your next action. You have access to the following actions:

                {actions}

                Examples:
            """).strip(),
            example_prompt=PromptTemplate.from_template("EXAMPLE START:\n{content}\nEND OF EXAMPLE"),
            example_selector=self.example_selector,
            suffix=textwrap.dedent("""
                Begin!

                Summary: {scene_summary}
                Obs: {action_description}
                {react_history}Act:
            """).strip(),
            input_variables=["character_summary", "scene_summary", "action_description", "react_history"],
            example_separator="\n\n",
        )

    @property
    def input_keys(self) -> List[str]:
        return ["scene_summary", "input_action"]

    @property
    def return_values(self) -> List[str]:
        return ["output_action"]

    def get_allowed_tools(self) -> Optional[List[str]]:
        return [t.name for t in self.tools]

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        return self.plan(intermediate_steps, callbacks, **kwargs)

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        react_history = ReActUtils.parse_react_history(intermediate_steps)
        prompt = self.template.format(
            character_summary=self.npc.sheet.summary,
            scene_summary=kwargs["scene_summary"],
            action_description=kwargs["input_action"].action_description,
            react_history="" if not react_history else (react_history + "\n"),
        )
        answer = self.llm(prompt)
        action = ReActUtils.extract_action(answer)
        tool = self.tool_lookup[action.tool]
        if isinstance(tool, RolePlayTool):
            return AgentFinish(
                return_values={"output_action": tool.role_play_action(action)},
                log=action.log
            )
        return action


class SceneSummarizer(BaseModel):
    llm: BaseLLM
    template = PromptTemplate.from_template(textwrap.dedent("""
        I found myself in the following situation:
        {summary}

        Then, the following events happened:
        {events}

        Please write a short paragraph, from my own point of view.
        It should summarize concisely the situation and events mentioned above.
    """).strip())

    def __call__(self, current_summary: str, new_events: Iterable[RolePlayAction]) -> str:
        prompt = self.template.format(
            summary=current_summary.strip(),
            events="\n".join([m.action_description.strip() for m in new_events])
        )
        return self.llm(prompt).strip()


class NpcScene:

    def __init__(self, agent: NpcAgent, scene_summarizer: SceneSummarizer, trigger: str = ""):
        self.agent = agent
        self.executor = AgentExecutor(agent=agent, tools=agent.tools, verbose=True)
        self.summary = f"{agent.npc.situation.rstrip('.').capitalize()}. {trigger.capitalize()}"
        self.scene_summarizer = scene_summarizer

    def add_to_summary(self, *new_events: RolePlayAction):
        self.summary = self.scene_summarizer(
            current_summary=self.summary,
            new_events=new_events,
        )

    def interact(self, input_action: RolePlayAction) -> RolePlayAction:
        inputs = dict(scene_summary=self.summary, input_action=input_action)
        output_action: RolePlayAction = self.executor(inputs)["output_action"]
        self.add_to_summary(input_action, output_action)
        return output_action.with_actor(self.agent.npc.sheet.name)

    def say(self, interlocutor: str, text: str) -> RolePlayAction:
        return self.interact(SpeechAction(actor=interlocutor, text=text))

    def finish_scene(self):
        self.agent.npc.knowledge_base.vectorstore.add_documents([
            KnowledgeDocTypes.scene(self.summary)
        ])
