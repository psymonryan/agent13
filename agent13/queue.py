"""Agent queue - priority queue for processing items."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ItemStatus(Enum):
    """Status of a queue item."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"


@dataclass
class QueueItem:
    """An item in the agent queue."""

    id: int
    text: str
    priority: bool = False
    interrupt: bool = False  # Interrupt level - triggers agent loop break
    kind: str = (
        "prompt"  # "prompt", "journal_last", "journal_all", "clear", "load", "retry"
    )
    status: ItemStatus = ItemStatus.PENDING
    data: dict = None  # Optional metadata (e.g. {"clear_widgets": True})


class AgentQueue:
    """Manages the queue of pending items for the agent to process.

    Items can be added with three priority levels:
    - Normal: appended to end of queue
    - High priority (!): inserted after interrupt items, before normal items
    - Interrupt (!!) inserted at front, triggers agent loop break

    Only one item is processed at a time.
    """

    def __init__(self):
        self.items: list[QueueItem] = []
        self.counter = 0
        self.current: Optional[QueueItem] = None

    def add(
        self,
        text: str,
        priority: bool = False,
        interrupt: bool = False,
        kind: str = "prompt",
        data: dict = None,
    ) -> int:
        """Add item to queue. Returns item ID.

        Args:
            text: The message text
            priority: If True, insert after interrupt items, before normal items
            interrupt: If True, insert at front and trigger agent loop break
            data: Optional metadata dict carried with the item

        Order (front to back): interrupt items -> priority items -> normal items
        """
        self.counter += 1
        item = QueueItem(
            id=self.counter,
            text=text,
            priority=priority or interrupt,
            interrupt=interrupt,
            kind=kind,
            data=data,
        )

        if interrupt:
            # Insert at front, after other interrupt items
            insert_at = 0
            for i, existing in enumerate(self.items):
                if existing.interrupt:
                    insert_at = i + 1
                else:
                    break
            self.items.insert(insert_at, item)
        elif priority:
            # Insert after all interrupt items, after existing priority items, before normal items
            insert_at = 0
            for i, existing in enumerate(self.items):
                if existing.interrupt or existing.priority:
                    insert_at = i + 1
                else:
                    break
            self.items.insert(insert_at, item)
        else:
            self.items.append(item)

        return item.id

    def get_next(self) -> Optional[QueueItem]:
        """Get next item to process. Returns None if queue is empty or item is running."""
        if self.current is None and self.items:
            self.current = self.items.pop(0)
            self.current.status = ItemStatus.RUNNING
        return self.current

    def complete_current(self):
        """Mark current item as complete and clear it."""
        if self.current:
            self.current.status = ItemStatus.COMPLETE
            self.current = None

    def remove(self, item_id: int) -> bool:
        """Remove item from queue by ID. Returns True if found and removed."""
        for i, item in enumerate(self.items):
            if item.id == item_id:
                del self.items[i]
                return True
        return False

    def remove_at(self, index: int) -> Optional[QueueItem]:
        """Remove item at index (1-based). Returns the removed item or None."""
        if 1 <= index <= len(self.items):
            return self.items.pop(index - 1)
        return None

    def set_priority(self, item_id: int, priority: bool) -> bool:
        """Change item priority. Returns True if item was found."""
        for i, item in enumerate(self.items):
            if item.id == item_id:
                if item.priority != priority:
                    # Remove and re-add with new priority (preserve interrupt flag)
                    del self.items[i]
                    item.priority = priority
                    self.add(item.text, priority=priority, interrupt=item.interrupt)
                return True
        return False

    def set_priority_at(self, index: int, priority: bool) -> bool:
        """Change priority of item at index (1-based). Returns True if successful."""
        if 1 <= index <= len(self.items):
            item = self.items[index - 1]
            if item.priority != priority:
                del self.items[index - 1]
                item.priority = priority
                self.add(item.text, priority=priority, interrupt=item.interrupt)
            return True
        return False

    @property
    def pending_count(self) -> int:
        """Number of pending items in queue."""
        return len(self.items)

    @property
    def has_priority(self) -> bool:
        """Whether there are any priority items pending."""
        return any(item.priority for item in self.items)

    @property
    def has_interrupt(self) -> bool:
        """Whether there are any interrupt-level items pending."""
        return any(item.interrupt for item in self.items)

    def pop_interrupt_items(self) -> list[QueueItem]:
        """Remove and return all interrupt items from front of queue."""
        interrupt_items = []
        while self.items and self.items[0].interrupt:
            interrupt_items.append(self.items.pop(0))
        return interrupt_items

    def pop_priority_items(self) -> list[QueueItem]:
        """Remove and return all priority items from front of queue."""
        priority_items = []
        while self.items and self.items[0].priority:
            priority_items.append(self.items.pop(0))
        return priority_items

    def list_items(self) -> list[QueueItem]:
        """Return list of pending items (not including currently running)."""
        return list(self.items)

    def clear(self) -> int:
        """Clear all pending items. Returns count of cleared items."""
        count = len(self.items)
        self.items.clear()
        return count
