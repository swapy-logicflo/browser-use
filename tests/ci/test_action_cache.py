from pydantic import create_model

from browser_use.agent.action_cache import build_action_cache
from browser_use.agent.views import AgentHistory, AgentHistoryList, AgentOutput, ActionResult, StepMetadata
from browser_use.browser.views import BrowserStateHistory
from browser_use.dom.views import DOMInteractedElement, NodeType
from browser_use.tools.registry.views import ActionModel
from browser_use.tools.service import Tools

URL = 'https://example.com'


def _step(model_output, result, interacted_element):
	return AgentHistory(
		model_output=model_output,
		result=result,
		state=BrowserStateHistory(url=URL, title='t', tabs=[], interacted_element=interacted_element),
		metadata=StepMetadata(step_start_time=0, step_end_time=1, step_number=1),
	)


def test_skips_an_action_with_no_field_set():
	"""Tools.registry.create_action_model() falls back to this empty model when zero actions are
	registered (service.py:528-529); its model_dump(exclude_unset=True) is always {}, which must not
	crash build_action_cache the way an unguarded `next(iter({}.items()))` would."""
	empty_action_model = create_model('EmptyActionModel', __base__=ActionModel)()
	assert empty_action_model.model_dump(exclude_unset=True) == {}

	output = AgentOutput.model_construct(next_goal='g', action=[empty_action_model])
	cache = build_action_cache('task', AgentHistoryList(history=[_step(output, [ActionResult()], [None])]))

	assert cache.actions == []


def test_records_a_real_click_action_with_its_resolved_element():
	action_model_cls = Tools(exclude_actions=[]).registry.create_action_model(include_actions=['click'])
	click = action_model_cls(click={'index': 5})
	assert click.model_dump(exclude_unset=True) == {'click': {'index': 5}}

	element = DOMInteractedElement(
		node_id=1,
		backend_node_id=1,
		frame_id=None,
		node_type=NodeType.ELEMENT_NODE,
		node_value='',
		node_name='BUTTON',
		attributes={'id': 'submit'},
		bounds=None,
		x_path="html/body/button[@id='submit']",
		element_hash=hash('submit'),
		ax_role='button',
		ax_name='Submit',
	)
	output = AgentOutput.model_construct(next_goal='Click submit', action=[click])
	cache = build_action_cache('task', AgentHistoryList(history=[_step(output, [ActionResult()], [element])]))

	assert len(cache.actions) == 1
	recorded = cache.actions[0]
	assert recorded.name == 'click'
	assert recorded.params == {}  # the index is stripped; it's a snapshot-local number, not a stable identity
	assert recorded.element is not None
	assert recorded.element.attributes == {'id': 'submit'}
	assert recorded.element.ax_name == 'Submit'
