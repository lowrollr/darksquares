


from typing import List, Tuple


def convert_coords_to_squares(coords: List[Tuple[int, int]]) -> List[int]:
    results = []
    for r, c in coords:
        results.append((r * 8) + c)
    return results


def convert_squares_to_coords(squares: List[int]) -> List[Tuple[int, int]]:
    results = []
    for sq in squares:
        r = sq // 8
        c = sq % 8
        results.append((r, c))
    return results

def opposite_square(sq: int) -> int:
    r = sq // 8
    c = sq % 8
    return ((7 - r) * 8) + (7 - c)

def opposite_rc(r: int, c: int) -> Tuple[int, int]:
    return (7 - r, 7 - c)