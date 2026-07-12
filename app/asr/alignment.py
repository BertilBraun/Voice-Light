from __future__ import annotations

from enum import StrEnum

from app.asr.normalization import normalized_word_pairs
from app.asr.transcript import Word
from app.frozen_base_config import FrozenBaseModel


class AlignmentOperation(StrEnum):
    EQUAL = "equal"
    SUBSTITUTE = "substitute"
    DELETE = "delete"
    INSERT = "insert"


class AlignedWord(FrozenBaseModel):
    reference: Word | None
    prediction: Word | None
    operation: AlignmentOperation
    reference_token: str | None
    prediction_token: str | None


class WordErrorCounts(FrozenBaseModel):
    substitutions: int
    insertions: int
    deletions: int
    reference_words: int

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            return 0.0
        return (self.substitutions + self.insertions + self.deletions) / self.reference_words


def align_words(
    reference_words: tuple[Word, ...], prediction_words: tuple[Word, ...]
) -> list[AlignedWord]:
    reference_pairs = normalized_word_pairs(reference_words)
    prediction_pairs = normalized_word_pairs(prediction_words)
    reference_tokens = [token for _word, token in reference_pairs]
    prediction_tokens = [token for _word, token in prediction_pairs]
    row_count = len(reference_tokens) + 1
    column_count = len(prediction_tokens) + 1
    costs = [[0 for _column in range(column_count)] for _row in range(row_count)]
    moves = [["" for _column in range(column_count)] for _row in range(row_count)]

    for row_index in range(1, row_count):
        costs[row_index][0] = row_index
        moves[row_index][0] = AlignmentOperation.DELETE.value
    for column_index in range(1, column_count):
        costs[0][column_index] = column_index
        moves[0][column_index] = AlignmentOperation.INSERT.value

    for row_index in range(1, row_count):
        for column_index in range(1, column_count):
            if reference_tokens[row_index - 1] == prediction_tokens[column_index - 1]:
                diagonal_cost = costs[row_index - 1][column_index - 1]
                diagonal_move = AlignmentOperation.EQUAL
            else:
                diagonal_cost = costs[row_index - 1][column_index - 1] + 1
                diagonal_move = AlignmentOperation.SUBSTITUTE
            delete_cost = costs[row_index - 1][column_index] + 1
            insert_cost = costs[row_index][column_index - 1] + 1
            best_cost = min(diagonal_cost, delete_cost, insert_cost)
            costs[row_index][column_index] = best_cost
            if diagonal_move == AlignmentOperation.EQUAL and best_cost == diagonal_cost:
                moves[row_index][column_index] = AlignmentOperation.EQUAL.value
            elif best_cost == delete_cost:
                moves[row_index][column_index] = AlignmentOperation.DELETE.value
            elif best_cost == insert_cost:
                moves[row_index][column_index] = AlignmentOperation.INSERT.value
            else:
                moves[row_index][column_index] = AlignmentOperation.SUBSTITUTE.value

    aligned_reversed: list[AlignedWord] = []
    row_index = len(reference_tokens)
    column_index = len(prediction_tokens)
    while row_index > 0 or column_index > 0:
        move = AlignmentOperation(moves[row_index][column_index])
        if move in {AlignmentOperation.EQUAL, AlignmentOperation.SUBSTITUTE}:
            reference_word, reference_token = reference_pairs[row_index - 1]
            prediction_word, prediction_token = prediction_pairs[column_index - 1]
            aligned_reversed.append(
                AlignedWord(
                    reference=reference_word,
                    prediction=prediction_word,
                    operation=move,
                    reference_token=reference_token,
                    prediction_token=prediction_token,
                )
            )
            row_index -= 1
            column_index -= 1
        elif move == AlignmentOperation.DELETE:
            reference_word, reference_token = reference_pairs[row_index - 1]
            aligned_reversed.append(
                AlignedWord(
                    reference=reference_word,
                    prediction=None,
                    operation=AlignmentOperation.DELETE,
                    reference_token=reference_token,
                    prediction_token=None,
                )
            )
            row_index -= 1
        elif move == AlignmentOperation.INSERT:
            prediction_word, prediction_token = prediction_pairs[column_index - 1]
            aligned_reversed.append(
                AlignedWord(
                    reference=None,
                    prediction=prediction_word,
                    operation=AlignmentOperation.INSERT,
                    reference_token=None,
                    prediction_token=prediction_token,
                )
            )
            column_index -= 1
    aligned_reversed.reverse()
    return aligned_reversed


def word_error_counts(alignment: list[AlignedWord]) -> WordErrorCounts:
    substitutions = sum(1 for item in alignment if item.operation == AlignmentOperation.SUBSTITUTE)
    insertions = sum(1 for item in alignment if item.operation == AlignmentOperation.INSERT)
    deletions = sum(1 for item in alignment if item.operation == AlignmentOperation.DELETE)
    reference_words = sum(1 for item in alignment if item.reference is not None)
    return WordErrorCounts(
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        reference_words=reference_words,
    )
