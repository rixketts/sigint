# class DarkObject:
#     """
#     A dark object in Ector's problem has two different
#     base probabilities, based on whether that object is 
#     is round or square
#     """
#     def __init__(self) -> None:
#         self.base_prob_round = 0.25

#         self.base_prob_square = 0.25
#         self.p_coef_square = 0.25


#     def get_probability_round(self, p: float) -> float:
#         return self.base_prob_round
    

#     def get_probability_square(self, p: float) -> float:
#         return self.base_prob_square + p * self.p_coef_square


class DarkObject:
    def __init__(self) -> None:
        self.base_prob = 0.25
        self.p_coef = 0.25


    def get_probability(self, p: float) -> float:
        """
        Get probability for shape-agnostic problem
        """
        return self.base_prob + p * self.p_coef


class RoundObject(DarkObject):
    def __init__(self) -> None:
        self.base_prob = 0.25

    
    def get_probability(self, p: float) -> float:
        """
        Get probability for shape-aware problem
        """
        return self.base_prob
    

class SquareObject(DarkObject):
    def __init__(self) -> None:
        self.base_prob = 0.25
        self.p_coef = 0.25


    def get_probability(self, p: float) -> float:
        """
        Get probability for shape-aware problem
        """
        return self.base_prob + p * self.p_coef
