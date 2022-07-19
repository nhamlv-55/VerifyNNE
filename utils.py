"""
Utility class to parse ACAS
"""
class CommaString(object):
    """ A full string separated by commas. """
    def __init__(self, text: str):
        self.text = text
        return

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text

    def has_next_comma(self) -> bool:
        return ',' in self.text

    def _read_next(self) -> str:
        """
        :return: the raw string of next token before comma
        """
        if self.has_next_comma():
            token, self.text = self.text.split(',', maxsplit=1)
        else:
            token, self.text = self.text, ''
        return token.strip()

    def read_next_as_int(self) -> int:
        return int(self._read_next())

    def read_next_as_float(self) -> float:
        return float(self._read_next())

    def read_next_as_bool(self) -> bool:
        """ Parse the next token before comma as boolean, 1/0 for true/false. """
        num = self.read_next_as_int()
        assert num == 1 or num == 0, f'The should-be-bool number is {num}.'
        return bool(num)
     