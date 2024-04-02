"""ECS Non-Primitive Data Types"""

from __future__ import annotations
from typing import Any


class TrieNode:
    _id: str
    nodes: dict[str, TrieNode]
    prev: TrieNode | None

    def __init__(self, _id: str):
        self._id = _id
        self.nodes = {}

    def get(self, _id: str) -> Any:
        """Get child node

        Returns:
            (TrieNode | None): Child node if found.
        """
        return self.nodes.get(_id, None)

    def set(self, _id: str, new_node: TrieNode) -> None:
        """Set Child Node

        Args:
            _id (str): Trie_id
            new_node (TrieNode): child node

        Raises:
            AttributeError: Attempting to overwrite existing node.
        """
        if self.get(_id):
            raise AttributeError(
                f"Attempting to overwrite existing node: {_id}."
            )
        new_node.prev = self
        self.nodes.update({_id: new_node})

    def display(self):
        """Display Trie From Node

        Returns:
            (str): multi-line str representing trie
        """
        nodes = self.nodes.items()
        if len(nodes) == 0:
            return self._id
        _display = f"{self._id} |"
        for _, v in nodes:
            next_node = v.display()

            _display = _display + f"\n\t->{next_node}"
        return _display
