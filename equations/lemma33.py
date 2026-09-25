from src.depth import DepthPoset, ShallowPair
from src.transpositions import Transposition
from src.equation import Equation
from functools import wraps


# class method wrappers
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


# set operations
def oplus(*args):
    res = set()
    for arg in args:
        res = res.symmetric_difference(arg)
    return res


# get set functions
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


def get_source_cell_filtration_index(source_cell, nodes: list[ShallowPair]) -> int:
    """
    """
    for node in nodes:
        if node.source[0] == source_cell:
            return node.birth_index
        if node.source[1] == source_cell:
            return node.death_index

def get_l_set(dp: DepthPoset, a: int | None=None, b: int | None=None, x: int | None=None, y: int | None=None):
    r"""
    $\mathcal{L} = \{(s, t)\in \text{BD}:\; f(t) < f(y)\}$
    """
    fy = get_source_cell_filtration_index(y, dp.nodes)
    return {node.source for node in dp.nodes if node.death_index < fy}

def get_m_set(dp: DepthPoset, a: int | None=None, b: int | None=None, x: int | None=None, y: int | None=None):
    r"""
    $\mathcal{M} = \{(s, t)\in \text{BD}:\; f(y) < f(t) < f(b)\}$
    """
    fy = get_source_cell_filtration_index(y, dp.nodes)
    fb = get_source_cell_filtration_index(b, dp.nodes)
    return {node.source for node in dp.nodes if (fy < node.death_index) & (node.death_index < fb)}

def get_b_set(dp: DepthPoset, a: int | None=None, b: int | None=None, x: int | None=None, y: int | None=None):
    r"""
    $\mathcal{B} = \{(s, t)\in \text{BD}:\; f(s) > f(x)\}$
    """
    fx = get_source_cell_filtration_index(x, dp.nodes)
    return {node.source for node in dp.nodes if node.birth_index > fx}

def get_n_set(dp: DepthPoset, a: int | None=None, b: int | None=None, x: int | None=None, y: int | None=None):
    r"""
    $\mathcal{N} = \{(s, t)\in \text{BD}:\; f(x) > f(s) > f(a)\}$
    """
    fx = get_source_cell_filtration_index(x, dp.nodes)
    fa = get_source_cell_filtration_index(a, dp.nodes)
    return {node.source for node in dp.nodes if (fx > node.birth_index) & (node.birth_index > fa)}



# birth-birth transposition
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


# death-death transpositions
class Eq28(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{at}(x, b)$
        """
        return get_succ1(dp_at, x, b)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{bt}(x, y) \oplus \{(a, y), (a, b)\}$
        """
        return oplus(
            get_succ1(dp_bt, x, y), 
            {(a, y), (a, b)}
        )
    
class Eq29(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{at}(a, y)$
        """
        return get_succ1(dp_at, a, y)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{bt}(a, b)$
        """
        return get_succ1(dp_bt, a, b)

class Eq30(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_1^\text{at}(x, b)$
        """
        return get_pred1(dp_at, x, b)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $[\text{Pred}_1^\text{bt}(a, b) \cap \mathcal{B}]$
        """
        return get_pred1(dp_bt, a, b) & get_b_set(dp_bt, a, b, x, y)

class Eq31(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_1^\text{at}(a, y)$
        """
        return get_pred1(dp_at, a, y)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_1^\text{bt}(x, y) \oplus \{(x, b)\} \oplus[\text{Pred}_1^\text{bt}(a, b) \cap \mathcal{N}]$
        """
        return oplus(
            get_pred1(dp_bt, x, y), 
            {(x, b)}, 
            get_pred1(dp_bt, a, b) & get_n_set(dp_bt, a, b, x, y)
        )


class Eq32(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{at}(x, b)$
        """
        return get_succ2(dp_at, x, b)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{bt}(x, y) \oplus \{(a, y)\} \oplus \text{Succ}_2^\text{bt}(a, b)$
        """
        return oplus(
            get_succ2(dp_bt, x, y), 
            {(a, y)}, 
            get_succ2(dp_bt, a, b), 
        )

class Eq33(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{at}(a, y)$
        """
        return get_succ2(dp_at, a, y)
    
    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{bt}(a, b)$
        """
        return get_succ2(dp_bt, a, b)

class Eq34(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_2^\text{at}(x, b)$
        """
        return get_pred2(dp_at, x, b)
    
    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_2^\text{bt}(x, y)$
        """
        return get_pred2(dp_bt, x, y)
    
class Eq35(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_2^\text{at}(a, y)$
        """
        return get_pred2(dp_at, a, y)
    
    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_2^\text{bt}(a, b) \oplus \{(x, b)\}$
        """
        return oplus(
            get_pred2(dp_bt, a, b), 
            {(x, b)}
        )

class Eq36(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{at}(x, y)$
        """
        return get_succ2(dp_at, x, y)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{bt}(x, y) \oplus \{(a, b)\} \oplus \text{Succ}_2^\text{bt}(a, b)$
        """
        return oplus(
            get_succ2(dp_bt, x, y), 
            {(a, b)}, 
            get_succ2(dp_bt, a, b), 
        )
    
class Eq37(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_2^\text{at}(a, b)$
        """
        return get_pred2(dp_at, a, b)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_2^\text{bt}(a, b) \oplus \{(x, y)\}$
        """
        return oplus(
            get_pred2(dp_bt, a, b), 
            {(x, y)}
        )


# birth-death transpositions
class Eq48(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{at}(a, x)$
        """
        return get_succ1(dp_at, a, x)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{bt}(a, b)
        """
        return get_succ1(dp_bt, a, b)
    
class Eq49(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{at}(b, y)$
        """
        return get_succ1(dp_at, b, y)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{bt}(x, y)$
        """
        return get_succ1(dp_bt, x, y)

class Eq50(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_1^\text{at}(a, x)$
        """
        return get_pred1(dp_at, a, x)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\{(t, x)| U_1^\text{bt}[t, x] = 1\}$
        """
        return {(t, x) for t, s in dp_bt._b0_set if s == x}

class Eq51(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_1^\text{at}(b, y)$
        """
        return get_pred1(dp_at, b, y)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_1^\text{bt}(x, y)$
        """
        return get_pred1(dp_bt, x, y)

class Eq52(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{at}(a, x)$
        """
        return get_succ2(dp_at, a, x)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{bt}(a, b)
        """
        return get_succ2(dp_bt, a, b)

class Eq53(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{at}(b, y)$
        """
        return get_succ2(dp_at, b, y)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{bt}(x, y)$
        """
        return get_succ2(dp_bt, x, y)

class Eq54(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_2^\text{at}(a, x)$
        """
        return get_pred2(dp_at, a, x)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_2^\text{bt}(a, b)$
        """
        return get_pred2(dp_bt, a, b)

class Eq55(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_2^\text{at}(b, y)$
        """
        return get_pred2(dp_at, b, y)

    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\{(b, s) | U_2^\text{bt}[b, s] = 1\}$
        """
        return {(b, s) for s, t in dp_at._b1_set if t == b}


# No switch, no nested cases
class EqSucc1ab(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{at}(a, b)$
        """
        return get_succ1(dp_at, a, b)
    
    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{bt}(a, b)$
        """
        return get_succ1(dp_bt, a, b)

class EqSucc1xy(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{at}(x, y)$
        """
        return get_succ1(dp_at, x, y)
    
    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_1^\text{bt}(x, y)$
        """
        return get_succ1(dp_bt, x, y)
    
class EqPred1ab(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_1^\text{at}(a, b)$
        """
        return get_pred1(dp_at, a, b)
    
    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_1^\text{bt}(a, b)$
        """
        return get_pred1(dp_bt, a, b)

class EqPred1xy(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_1^\text{at}(x, y)$
        """
        return get_pred1(dp_at, x, y)
    
    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_1^\text{bt}(x, y)$
        """
        return get_pred1(dp_bt, x, y)
    
class EqSucc2ab(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{at}(a, b)$
        """
        return get_succ2(dp_at, a, b)
    
    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{bt}(a, b)$
        """
        return get_succ2(dp_bt, a, b)

class EqSucc2xy(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{at}(x, y)$
        """
        return get_succ2(dp_at, x, y)
    
    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Succ}_2^\text{bt}(x, y)$
        """
        return get_succ2(dp_bt, x, y)

class EqPred2ab(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_2^\text{at}(a, b)$
        """
    
    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_2^\text{bt}(a, b)$
        """
        return get_pred2(dp_bt, a, b)

class EqPred2xy(Equation):
    @resolve_eq_params
    def left(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_2^\text{at}(x, y)$
        """
        return get_pred2(dp_at, x, y)
    
    @resolve_eq_params
    def right(self, dp_bt, dp_at, x, y, a, b):
        r"""
        $\text{Pred}_2^\text{bt}(x, y)$
        """
        return get_pred2(dp_bt, x, y)