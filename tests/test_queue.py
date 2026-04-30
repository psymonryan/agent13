"""Tests for agent.queue module."""

from agent13 import AgentQueue, QueueItem, ItemStatus


class TestQueueItem:
    """Tests for QueueItem dataclass."""

    def test_create_queue_item(self):
        """Should create a queue item with default values."""
        item = QueueItem(id=1, text="hello")
        assert item.id == 1
        assert item.text == "hello"
        assert item.priority is False
        assert item.status == ItemStatus.PENDING

    def test_create_priority_item(self):
        """Should create a priority queue item."""
        item = QueueItem(id=1, text="urgent", priority=True)
        assert item.priority is True


class TestItemStatus:
    """Tests for ItemStatus enum."""

    def test_status_values(self):
        """Status values should match expected strings."""
        assert ItemStatus.PENDING.value == "pending"
        assert ItemStatus.RUNNING.value == "running"
        assert ItemStatus.COMPLETE.value == "complete"


class TestAgentQueue:
    """Tests for AgentQueue class."""

    def test_create_queue(self):
        """Should create an empty queue."""
        q = AgentQueue()
        assert q.items == []
        assert q.counter == 0
        assert q.current is None

    def test_add_item(self):
        """Should add item and return ID."""
        q = AgentQueue()
        item_id = q.add("hello")
        assert item_id == 1
        assert q.pending_count == 1

    def test_add_multiple_items(self):
        """Should add multiple items with incrementing IDs."""
        q = AgentQueue()
        id1 = q.add("first")
        id2 = q.add("second")
        id3 = q.add("third")
        assert id1 == 1
        assert id2 == 2
        assert id3 == 3
        assert q.pending_count == 3

    def test_add_priority_item(self):
        """Priority items should be inserted before normal items."""
        q = AgentQueue()
        q.add("normal 1")
        q.add("normal 2")
        q.add("priority 1", priority=True)

        items = q.list_items()
        assert items[0].text == "priority 1"
        assert items[1].text == "normal 1"
        assert items[2].text == "normal 2"

    def test_add_multiple_priority_items(self):
        """Multiple priority items should maintain order."""
        q = AgentQueue()
        q.add("normal")
        q.add("priority 1", priority=True)
        q.add("priority 2", priority=True)

        items = q.list_items()
        assert items[0].text == "priority 1"
        assert items[1].text == "priority 2"
        assert items[2].text == "normal"

    def test_get_next(self):
        """Should get next item and mark as running."""
        q = AgentQueue()
        q.add("item 1")
        q.add("item 2")

        item = q.get_next()
        assert item.text == "item 1"
        assert item.status == ItemStatus.RUNNING
        assert q.current == item
        assert q.pending_count == 1

    def test_get_next_empty_queue(self):
        """Should return None if queue is empty."""
        q = AgentQueue()
        assert q.get_next() is None

    def test_get_next_while_running(self):
        """Should return current item if already running."""
        q = AgentQueue()
        q.add("item 1")
        q.add("item 2")

        item1 = q.get_next()
        item2 = q.get_next()  # Should return same item

        assert item1 == item2
        assert q.pending_count == 1

    def test_complete_current(self):
        """Should mark current item as complete."""
        q = AgentQueue()
        q.add("item 1")
        item = q.get_next()

        q.complete_current()

        assert item.status == ItemStatus.COMPLETE
        assert q.current is None

    def test_remove_by_id(self):
        """Should remove item by ID."""
        q = AgentQueue()
        id1 = q.add("item 1")
        q.add("item 2")

        result = q.remove(id1)

        assert result is True
        assert q.pending_count == 1
        assert q.list_items()[0].text == "item 2"

    def test_remove_by_id_not_found(self):
        """Should return False if ID not found."""
        q = AgentQueue()
        q.add("item 1")

        result = q.remove(999)

        assert result is False

    def test_remove_at(self):
        """Should remove item at 1-based index."""
        q = AgentQueue()
        q.add("item 1")
        q.add("item 2")
        q.add("item 3")

        removed = q.remove_at(2)

        assert removed.text == "item 2"
        assert q.pending_count == 2

    def test_remove_at_invalid_index(self):
        """Should return None for invalid index."""
        q = AgentQueue()
        q.add("item 1")

        assert q.remove_at(0) is None
        assert q.remove_at(2) is None

    def test_set_priority(self):
        """Should change item priority."""
        q = AgentQueue()
        id1 = q.add("normal")
        q.add("other")

        q.set_priority(id1, priority=True)

        items = q.list_items()
        assert items[0].text == "normal"
        assert items[0].priority is True

    def test_set_priority_at(self):
        """Should change priority by index."""
        q = AgentQueue()
        q.add("item 1")
        q.add("item 2")

        q.set_priority_at(2, priority=True)

        items = q.list_items()
        assert items[0].text == "item 2"
        assert items[0].priority is True

    def test_pending_count(self):
        """Should return count of pending items."""
        q = AgentQueue()
        assert q.pending_count == 0

        q.add("item 1")
        assert q.pending_count == 1

        q.add("item 2")
        assert q.pending_count == 2

    def test_has_priority(self):
        """Should return True if any priority items exist."""
        q = AgentQueue()
        assert q.has_priority is False

        q.add("normal")
        assert q.has_priority is False

        q.add("priority", priority=True)
        assert q.has_priority is True

    def test_pop_priority_items(self):
        """Should remove and return all priority items."""
        q = AgentQueue()
        q.add("priority 1", priority=True)
        q.add("priority 2", priority=True)
        q.add("normal")

        priority_items = q.pop_priority_items()

        assert len(priority_items) == 2
        assert q.pending_count == 1
        assert q.has_priority is False

    def test_list_items(self):
        """Should return list of pending items."""
        q = AgentQueue()
        q.add("item 1")
        q.add("item 2")

        items = q.list_items()

        assert len(items) == 2
        assert items[0].text == "item 1"

    def test_clear(self):
        """Should clear all pending items."""
        q = AgentQueue()
        q.add("item 1")
        q.add("item 2")

        count = q.clear()

        assert count == 2
        assert q.pending_count == 0
