"""
ACOC - Pruning Manager
======================
Gère la suppression et consolidation des blocs inutilisés.
"""

from typing import Dict, List, Tuple, Optional, Set

from ..config import SystemConfig, TaskBlock


class PruningManager:
    """
    Gère la suppression et consolidation des blocs inutilisés.
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.pruning_history: List[Tuple[int, str, str]] = []

    def identify_unused_blocks(
        self,
        task_blocks: Dict[str, TaskBlock],
        current_cycle: int,
        protected_blocks: Set[str] = None
    ) -> List[str]:
        """
        Identifie les blocs qui n'ont pas été utilisés depuis longtemps.
        """
        if protected_blocks is None:
            protected_blocks = set()

        unused = []

        for block_id, block in task_blocks.items():
            if block_id in protected_blocks:
                continue

            cycles_since_use = current_cycle - block.last_used_cycle

            if (cycles_since_use > self.config.prune_unused_after_cycles and
                block.usage_count < current_cycle * 0.1):
                unused.append(block_id)

        return unused

    def find_similar_blocks(
        self,
        task_blocks: Dict[str, TaskBlock]
    ) -> List[Tuple[str, str, float]]:
        """
        Trouve les paires de blocs suffisamment similaires pour être fusionnés.
        """
        similar_pairs = []
        block_ids = list(task_blocks.keys())

        for i, id1 in enumerate(block_ids):
            for id2 in block_ids[i + 1:]:
                block1 = task_blocks[id1]
                block2 = task_blocks[id2]

                if block1.task_type != block2.task_type:
                    continue

                similarity = self._compute_block_similarity(block1, block2)

                if similarity > self.config.consolidation_similarity_threshold:
                    similar_pairs.append((id1, id2, similarity))

        return similar_pairs

    def _compute_block_similarity(
        self,
        block1: TaskBlock,
        block2: TaskBlock
    ) -> float:
        """Calcule la similarité entre deux blocs."""
        size_ratio = min(block1.num_params, block2.num_params) / \
                     max(block1.num_params, block2.num_params)

        total_usage = block1.usage_count + block2.usage_count
        if total_usage > 0:
            usage_balance = 1 - abs(block1.usage_count - block2.usage_count) / total_usage
        else:
            usage_balance = 1.0

        type_match = 1.0 if block1.task_type == block2.task_type else 0.5

        return size_ratio * usage_balance * type_match

    def prune_block(
        self,
        task_blocks: Dict[str, TaskBlock],
        block_id: str,
        current_cycle: int
    ) -> bool:
        """Supprime un bloc."""
        if block_id in task_blocks:
            del task_blocks[block_id]
            self.pruning_history.append((current_cycle, block_id, "pruned"))
            return True
        return False

    def consolidate_blocks(
        self,
        task_blocks: Dict[str, TaskBlock],
        block_id_1: str,
        block_id_2: str,
        current_cycle: int
    ) -> Optional[str]:
        """Fusionne deux blocs en un seul."""
        if block_id_1 not in task_blocks or block_id_2 not in task_blocks:
            return None

        block1 = task_blocks[block_id_1]
        block2 = task_blocks[block_id_2]

        if block1.usage_count >= block2.usage_count:
            survivor_id, remove_id = block_id_1, block_id_2
            survivor = block1
        else:
            survivor_id, remove_id = block_id_2, block_id_1
            survivor = block2

        survivor.usage_count += task_blocks[remove_id].usage_count

        del task_blocks[remove_id]
        self.pruning_history.append((current_cycle, remove_id, f"merged_into_{survivor_id}"))

        return survivor_id

    def run_maintenance(
        self,
        task_blocks: Dict[str, TaskBlock],
        current_cycle: int,
        protected_blocks: Set[str] = None
    ) -> Dict[str, List[str]]:
        """Exécute un cycle complet de maintenance."""
        if protected_blocks is None:
            protected_blocks = set()

        actions = {"pruned": [], "consolidated": []}

        # 1. Identifier et supprimer les blocs inutilisés
        unused = self.identify_unused_blocks(task_blocks, current_cycle, protected_blocks)
        for block_id in unused:
            if self.prune_block(task_blocks, block_id, current_cycle):
                actions["pruned"].append(block_id)

        # 2. Consolider les blocs similaires (max 1 par cycle)
        similar = self.find_similar_blocks(task_blocks)
        similar = [(a, b, s) for a, b, s in similar
                   if a not in protected_blocks and b not in protected_blocks]

        if similar:
            id1, id2, _ = similar[0]
            survivor = self.consolidate_blocks(task_blocks, id1, id2, current_cycle)
            if survivor:
                removed = id1 if survivor == id2 else id2
                actions["consolidated"].append(f"{removed} → {survivor}")

        return actions
