"""Distill an agent run into an element-level action log that a later run can replay without the LLM.

An action's `index` is only valid for the DOM snapshot it was chosen from, so every action is stored
with the identity of the element it resolved to at that moment (attributes, xpath, text, ax role/name).
Deciding which actions are worth replaying is left to the consumer; this log records what happened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from browser_use.agent.views import AgentHistoryList
from browser_use.dom.views import DOMInteractedElement
from browser_use.tokens.views import UsageSummary


class CachedElement(BaseModel):
	tag: str
	attributes: dict[str, str]
	x_path: str
	element_hash: int
	text: str | None = None
	ax_role: str | None = None
	ax_name: str | None = None

	@classmethod
	def from_interacted(cls, element: DOMInteractedElement) -> CachedElement:
		return cls(
			tag=element.node_name.lower(),
			attributes=element.attributes or {},
			x_path=element.x_path,
			element_hash=element.element_hash,
			text=element.text,
			ax_role=element.ax_role,
			ax_name=element.ax_name,
		)


class CachedAction(BaseModel):
	step_number: int
	name: str
	params: dict[str, Any]
	element: CachedElement | None
	url: str
	goal: str | None
	# The agent's own evaluation of this step, written at the start of the next one ("... Verdict: Success").
	outcome: str | None = None
	error: str | None
	is_done: bool


class ActionCache(BaseModel):
	task: str
	start_url: str | None
	actions: list[CachedAction]
	llm_calls: int
	duration_seconds: float
	usage: UsageSummary | None

	def save(self, file_path: str | Path) -> None:
		path = Path(file_path)
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(self.model_dump_json(indent=2), encoding='utf-8')

	@classmethod
	def load(cls, file_path: str | Path) -> ActionCache:
		return cls.model_validate_json(Path(file_path).read_text(encoding='utf-8'))


def build_action_cache(task: str, history: AgentHistoryList) -> ActionCache:
	actions: list[CachedAction] = []
	steps = history.history
	for position, step in enumerate(steps, start=1):
		if step.model_output is None:
			continue
		step_number = step.metadata.step_number if step.metadata else position
		next_output = steps[position].model_output if position < len(steps) else None
		outcome = next_output.evaluation_previous_goal if next_output else None
		elements = step.state.interacted_element or []
		# multi_act stops early when the page changes, so only actions that produced a result were executed.
		for i, (action, result) in enumerate(zip(step.model_output.action, step.result)):
			name, params = next(iter(action.model_dump(exclude_unset=True).items()))
			element = elements[i] if i < len(elements) else None
			actions.append(
				CachedAction(
					step_number=step_number,
					name=name,
					params={key: value for key, value in (params or {}).items() if key != 'index'},
					element=CachedElement.from_interacted(element) if element else None,
					url=step.state.url,
					goal=step.model_output.next_goal,
					outcome=outcome,
					error=result.error,
					is_done=bool(result.is_done),
				)
			)

	return ActionCache(
		task=task,
		start_url=history.history[0].state.url if history.history else None,
		actions=actions,
		# The token tracker also sees invocations outside the step loop, so prefer its count.
		llm_calls=history.usage.entry_count if history.usage else len(history.history),
		duration_seconds=history.total_duration_seconds(),
		usage=history.usage,
	)
