"""Tests for queue priority ID preservation bug.

Bug: set_priority() and set_priority_at() change item IDs when reordering.

When set_priority() or set_priority_at() is called to change an item's priority,
they:
1. Delete the original item from the list
2. Call self.add() which creates a NEW QueueItem with a NEW ID

This means the original ID is lost. Any code tracking items by ID will lose
its reference after a priority change.

Example:
    queue = AgentQueue()
    item_id = queue.add("hello")  # Returns 1
    queue.set_priority(item_id, priority=True)
    # Now item has ID 2, not 1!
"""

from agent13.queue import AgentQueue


class TestSetPriorityPreservesID:
    """Verify that set_priority() preserves the item's ID."""

    def test_set_priority_preserves_item_id(self):
        """set_priority() should NOT change the item's ID.

        Currently, set_priority() deletes the item and calls self.add()
        which creates a new QueueItem with counter+1, changing the ID.
        """
        queue = AgentQueue()

        # Add an item - it gets ID 1
        original_id = queue.add("hello", priority=False)
        assert original_id == 1

        # Change priority from False to True
        result = queue.set_priority(original_id, priority=True)
        assert result is True  # Item was found

        # BUG: The item's ID should still be 1, but it's now 2
        items = queue.list_items()
        assert len(items) == 1, f"Expected 1 item, got {len(items)}"

        item = items[0]
        assert item.priority is True, "Priority should have been changed"
        assert item.id == original_id, (
            f"Item ID should be preserved. "
            f"Expected {original_id}, got {item.id}. "
            f"set_priority() calls self.add() which assigns a new ID."
        )

    def test_set_priority_does_not_increment_counter(self):
        """set_priority() should not increment the queue's counter.

        Since no new item is being added, the counter should stay the same.
        """
        queue = AgentQueue()

        original_id = queue.add("hello", priority=False)
        counter_before = queue.counter

        queue.set_priority(original_id, priority=True)

        # BUG: counter increments from 1 to 2
        assert queue.counter == counter_before, (
            f"Counter should not change when changing priority. "
            f"Was {counter_before}, now {queue.counter}. "
            f"set_priority() calls self.add() which increments counter."
        )

    def test_set_priority_same_priority_is_noop(self):
        """set_priority() with same priority should not change anything.

        This case is handled correctly - no deletion/re-addition occurs.
        """
        queue = AgentQueue()

        original_id = queue.add("hello", priority=False)
        counter_before = queue.counter

        # Setting same priority should be a no-op
        result = queue.set_priority(original_id, priority=False)
        assert result is True

        # Counter should not change
        assert queue.counter == counter_before

        # ID should be preserved (this works even with the bug)
        items = queue.list_items()
        assert items[0].id == original_id


class TestSetPriorityAtPreservesID:
    """Verify that set_priority_at() preserves the item's ID."""

    def test_set_priority_at_preserves_item_id(self):
        """set_priority_at() should NOT change the item's ID.

        Currently, set_priority_at() deletes the item and calls self.add()
        which creates a new QueueItem with counter+1, changing the ID.
        """
        queue = AgentQueue()

        # Add items
        first_id = queue.add("first", priority=False)
        _second_id = queue.add("second", priority=False)  # noqa: F841

        # Change priority of first item (index 1)
        result = queue.set_priority_at(1, priority=True)
        assert result is True

        # Find the item with priority=True
        items = queue.list_items()
        priority_items = [i for i in items if i.priority]
        assert len(priority_items) == 1, "Should have exactly one priority item"

        item = priority_items[0]
        assert item.priority is True

        # BUG: The item's ID should still be first_id, but it's now 3
        assert item.id == first_id, (
            f"Item ID should be preserved. "
            f"Expected {first_id}, got {item.id}. "
            f"set_priority_at() calls self.add() which assigns a new ID."
        )

    def test_set_priority_at_does_not_increment_counter(self):
        """set_priority_at() should not increment the queue's counter."""
        queue = AgentQueue()

        queue.add("first", priority=False)
        queue.add("second", priority=False)
        counter_before = queue.counter

        queue.set_priority_at(1, priority=True)

        # BUG: counter increments
        assert queue.counter == counter_before, (
            f"Counter should not change when changing priority. "
            f"Was {counter_before}, now {queue.counter}. "
            f"set_priority_at() calls self.add() which increments counter."
        )


class TestPriorityReordering:
    """Test that priority reordering works correctly."""

    def test_priority_item_moves_to_correct_position(self):
        """Priority items should be before normal items."""
        queue = AgentQueue()

        # Add normal items
        normal_id1 = queue.add("normal1", priority=False)
        _normal_id2 = queue.add("normal2", priority=False)  # noqa: F841

        # Make first item priority
        queue.set_priority(normal_id1, priority=True)

        items = queue.list_items()
        assert len(items) == 2

        # Priority item should come first
        assert items[0].priority is True
        assert items[0].text == "normal1"
        assert items[1].priority is False
        assert items[1].text == "normal2"
