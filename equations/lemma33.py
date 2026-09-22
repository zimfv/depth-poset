from src.depth import DepthPoset, ShallowPair
from src.transpositions import Transposition
from src.equation import Equation
from functools import wraps


def resolve_eq_params(func):
    @wraps(func)
    def wrapper(self, transposition: Transposition, dp_bt: DepthPoset | None = None, dp_at: DepthPoset | None = None, 
                x: int | None = None, y: int | None = None, a: int | None = None, b: int | None = None, **kwargs,):
        if dp_bt is None:
            dp_bt = transposition.dp

        if dp_at is None:
            dp_at = transposition.next_depth_poset()

        if any(v is None for v in (x, y, a, b)):
            x, y, a, b = transposition.get_xyab()

        return func(self, dp_bt=dp_bt, dp_at=dp_at, x=x, y=y, a=a, b=b)

    return wrapper


def set_tuples_int(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        res = {(int(a), int(b)) for a, b in func(*args, **kwargs)}
        return res
    return wrapper


def oplus(*args):
    res = set()
    for arg in args:
        res = res.symmetric_difference(arg)
    return res



def get_succ1(dp: DepthPoset, a: int, b: int):
    root = [node for node in dp.nodes if node.source == (a, b)][0]
    #return {node.source for node in dp.get_column_bottom_to_top_reduction().get_succesors(root).nodes}
    return {node.source for node in dp.get_succ1(root)}
    
def get_pred1(dp: DepthPoset, a: int, b: int):
    root = [node for node in dp.nodes if node.source == (a, b)][0]
    #return {node.source for node in dp.get_column_bottom_to_top_reduction().get_predecessors(root).nodes}
    return {node.source for node in dp.get_pred1(root)}
    
def get_succ2(dp: DepthPoset, a: int, b: int):
    root = [node for node in dp.nodes if node.source == (a, b)][0]
    #return {node.source for node in dp.get_row_left_to_right_reduction().get_succesors(root).nodes}
    return {node.source for node in dp.get_succ2(root)}
    
def get_pred2(dp: DepthPoset, a: int, b: int):
    root = [node for node in dp.nodes if node.source == (a, b)][0]
    #return {node.source for node in dp.get_row_left_to_right_reduction().get_predecessors(root).nodes}
    return {node.source for node in dp.get_pred2(root)}

def get_l_set(dp: DepthPoset, a: int | None=None, b: int | None=None, x: int | None=None, y: int | None=None):
    r"""
    $\mathcal{L} = \{(s, t)\in \text{BD}:\; f(t) < f(y)\}$
    """
    fy = None
    for node in dp.nodes:
        if node.source[0] == y:
            fy = node.birth_index
            break
        if node.source[1] == y:
            fy = node.death_index
            break
                
    return {node.source for node in dp.nodes if node.death_index < fy}


def get_m_set(dp: DepthPoset, a: int | None=None, b: int | None=None, x: int | None=None, y: int | None=None):
    r"""
    $\mathcal{M} = \{(s, t)\in \text{BD}:\; f(y) < f(t) < f(b)\}$
    """
    fy = None
    fb = None
    for node in dp.nodes:
        if node.source[0] == y:
            fy = node.birth_index
            break
        if node.source[1] == y:
            fy = node.death_index
            break
    for node in dp.nodes:
        if node.source[0] == b:
            fb = node.birth_index
            break
        if node.source[1] == b:
            fb = node.death_index
            break
    return {node.source for node in dp.nodes if (fy < node.death_index) & (node.death_index < fb)}

    
    

class Eq08(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{at}(a, y)$
        """
        return get_succ1(dp_at, a, y)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{bt}(x, y) \oplus \{(x, b)\} \oplus \text{Succ}_1^\text{bt}(a, b)$
        """
        return oplus(
            get_succ1(dp_bt, x, y), 
            {(x, b)}, 
            get_succ1(dp_bt, a, b), 
        )


class Eq09(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{at}(x, b)$
        """
        return get_succ1(dp_at, x, b)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{bt}(a, b)$
        """
        return get_succ1(dp_bt, a, b)

    
class Eq10(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_1^\text{at}(a, y)$
        """
        return get_pred1(dp_at, a, y)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_1^\text{bt}(x, y)$
        """
        return get_pred1(dp_bt, x, y)


class Eq11(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_1^\text{at}(x, b)$
        """
        return get_pred1(dp_at, x, b)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_1^\text{bt}(a, b) \oplus \{(a, y)\}$
        """
        return oplus(
            get_pred1(dp_bt, a, b), 
            {(a, y)}
        )


class Eq12(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{at}(a, y)$
        """
        return get_succ2(dp_at, a, y)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{bt}(x, y) \oplus \{(x, b), (a, b)\}$
        """
        return oplus(
            get_succ2(dp_bt, x, y), 
            {(x, b), (a, b)}
        )

class Eq13(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{at}(x, b)$
        """
        return get_succ2(dp_at, x, b)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{bt}(a, b)$
        """
        return get_succ2(dp_bt, a, b)


class Eq14(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_2^\text{at}(a, y)$
        """
        return get_pred2(dp_at, a, y)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $[\text{Pred}_2^\text{bt}(a, b) \cap \mathcal{L}]$
        """
        return get_pred2(dp_bt, a, b) & get_l_set(dp_bt, a, b, x, y)


class Eq15(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_2^\text{at}(x, b)$
        """
        return get_pred2(dp_at, x, b)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_2^\text{bt}(x, y) \oplus \{(a, y)\} \oplus [\text{Pred}_2^\text{bt}(a, b) \cap \mathcal{M}]$
        """
        return oplus(
            get_pred2(dp_bt, x, y), 
            {(a, y)}, 
            get_pred2(dp_bt, a, b) & get_m_set(dp_bt, a, b, x, y)
        )


class Eq16(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{at}(x, y)$
        """
        return get_succ1(dp_at, x, y)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{bt}(x, y) \oplus \{(a, b)\} \oplus \text{Succ}_1^\text{bt}(a, b)$
        """
        return oplus(
            get_succ1(dp_bt, x, y), 
            {(a, b)}, 
            get_succ1(dp_bt, a, b)
        )

class Eq17(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_1^\text{at}(a, b)$
        """
        return get_pred1(dp_at, a, b)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_1^\text{bt}(a, b) \oplus \{(x, y)\}$
        """
        return oplus(
            get_pred1(dp_bt, a, b), 
            {(x, y)}
        )