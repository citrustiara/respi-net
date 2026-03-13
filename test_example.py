import pytest
from example import Solution
class TestExample:
    @pytest.fixture
    def solution_maker(self):
        return Solution()
    def test_minOperations(self, solution_maker):
        assert solution_maker.minOperations('11') == 1
        assert solution_maker.minOperations('10') == 0
        assert solution_maker.minOperations('111111') == 3
        assert solution_maker.minOperations('0') == 0
        assert solution_maker.minOperations('1010101010') == 0
        assert solution_maker.minOperations('10000001') == 4
    def test_myAtoi(self, solution_maker):
        assert solution_maker.myAtoi('11') == 11
        assert solution_maker.myAtoi('      +11') == 11
        assert solution_maker.myAtoi('       -11') == -11
        assert solution_maker.myAtoi('      00011') == 11
        assert solution_maker.myAtoi('      0-11') == 0
        assert solution_maker.myAtoi('      0 11') == 0, "powinno tu byc zero"

