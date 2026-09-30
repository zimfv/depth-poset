class Equation:
    def left(self, *args, **kwargs):
        """
        0
        """
        return 0

    def right(self, *args, **kwargs):
        """
        0
        """
        return 0

    @property
    def left_str(self):
        return self.left.__doc__.replace('$', '').replace('\n', ' ').strip()

    @property
    def right_str(self):
        return self.right.__doc__.replace('$', '').replace('\n', ' ').strip()

    def is_correct(self, *args, **kwargs):
        """
        """
        return self.left(*args, **kwargs) == self.right(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.is_correct(*args, **kwargs)

    def __str__(self):
        """
        """
        return f'${self.left_str} = {self.right_str}$'

    def __repr__(self):
        return f'Eq({self.__str__()[1:-1]})'

    def __hash__(self):
        return hash(type(self))
    
    def __eq__(self, other):
        return type(self) is type(other)