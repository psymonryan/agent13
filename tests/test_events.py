"""Tests for agent.events module."""

from agent13 import AgentEvent, AgentEventData


class TestAgentEvent:
    """Tests for AgentEvent enum."""

    def test_event_types_exist(self):
        """All expected event types should exist."""
        assert hasattr(AgentEvent, "STARTED")
        assert hasattr(AgentEvent, "STOPPED")
        assert hasattr(AgentEvent, "QUEUE_UPDATE")
        assert hasattr(AgentEvent, "USER_MESSAGE")
        assert hasattr(AgentEvent, "ASSISTANT_TOKEN")
        assert hasattr(AgentEvent, "ASSISTANT_REASONING")
        assert hasattr(AgentEvent, "ASSISTANT_COMPLETE")
        assert hasattr(AgentEvent, "TOOL_CALL")
        assert hasattr(AgentEvent, "TOOL_RESULT")
        assert hasattr(AgentEvent, "STATUS_CHANGE")
        assert hasattr(AgentEvent, "ERROR")
        assert hasattr(AgentEvent, "MODEL_CHANGE")

    def test_event_values(self):
        """Event values should match their names."""
        assert AgentEvent.STARTED.value == "started"
        assert AgentEvent.STOPPED.value == "stopped"
        assert AgentEvent.QUEUE_UPDATE.value == "queue_update"
        assert AgentEvent.ASSISTANT_TOKEN.value == "assistant_token"


class TestAgentEventData:
    """Tests for AgentEventData dataclass."""

    def test_create_event_data(self):
        """Should create event data with event and optional data dict."""
        event_data = AgentEventData(event=AgentEvent.ASSISTANT_TOKEN)
        assert event_data.event == AgentEvent.ASSISTANT_TOKEN
        assert event_data.data == {}

    def test_create_event_data_with_data(self):
        """Should create event data with data dict."""
        event_data = AgentEventData(
            event=AgentEvent.ASSISTANT_TOKEN, data={"text": "hello"}
        )
        assert event_data.event == AgentEvent.ASSISTANT_TOKEN
        assert event_data.data == {"text": "hello"}

    def test_text_property(self):
        """text property should return data['text']."""
        event_data = AgentEventData(
            event=AgentEvent.ASSISTANT_TOKEN, data={"text": "hello"}
        )
        assert event_data.text == "hello"

    def test_text_property_missing(self):
        """text property should return None if not in data."""
        event_data = AgentEventData(event=AgentEvent.STARTED)
        assert event_data.text is None

    def test_name_property(self):
        """name property should return data['name'] (for tool events)."""
        event_data = AgentEventData(
            event=AgentEvent.TOOL_CALL, data={"name": "square_number"}
        )
        assert event_data.name == "square_number"

    def test_status_property(self):
        """status property should return data['status']."""
        event_data = AgentEventData(
            event=AgentEvent.STATUS_CHANGE, data={"status": "processing"}
        )
        assert event_data.status == "processing"

    def test_model_property(self):
        """model property should return data['model']."""
        event_data = AgentEventData(
            event=AgentEvent.MODEL_CHANGE, data={"model": "devstral"}
        )
        assert event_data.model == "devstral"

    def test_count_property(self):
        """count property should return data['count']."""
        event_data = AgentEventData(event=AgentEvent.QUEUE_UPDATE, data={"count": 5})
        assert event_data.count == 5

    def test_message_property(self):
        """message property should return data['message'] (for errors)."""
        event_data = AgentEventData(
            event=AgentEvent.ERROR, data={"message": "Something went wrong"}
        )
        assert event_data.message == "Something went wrong"

    def test_exception_property(self):
        """exception property should return data['exception']."""
        exc = ValueError("test error")
        event_data = AgentEventData(
            event=AgentEvent.ERROR, data={"message": "Error", "exception": exc}
        )
        assert event_data.exception is exc
