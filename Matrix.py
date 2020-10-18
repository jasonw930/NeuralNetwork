import random


class Matrix:
    @staticmethod
    def from_list(l):
        m = Matrix(len(l), 1)
        for i in range(len(l)):
            m.data[i][0] = l[i]
        return m

    @staticmethod
    def to_list(m):
        return [i for r in m.data for i in r]

    @staticmethod
    def from_str(string):
        val = string.split("\n")
        result = Matrix(*tuple(int(s) for s in val[0].split()))
        result.data = [[float(s) for s in line.split()] for line in val[1:]]
        return result

    @staticmethod
    def copy(m):
        result = Matrix(m.rows, m.cols)
        for r in range(m.rows):
            for c in range(m.cols):
                result.data[r][c] = m.data[r][c]
        return result

    @staticmethod
    def transpose(m):
        result = Matrix(m.cols, m.rows)
        for r in range(m.rows):
            for c in range(m.cols):
                result.data[c][r] = m.data[r][c]
        return result

    @staticmethod
    def matrix_product(m1, m2):
        if m1.cols != m2.rows:
            raise ValueError("Matrix operation error")
        depth = m1.cols
        result = Matrix(m1.rows, m2.cols)
        for r in range(m1.rows):
            for c in range(m2.cols):
                sum = 0
                for i in range(depth):
                    sum += m1.data[r][i] * m2.data[i][c]
                result.data[r][c] = sum
        return result

    @staticmethod
    def collapse(m):
        result = 0
        for r in range(m.rows):
            if m.data[r][0] > m.data[result][0]:
                result = r
        return result

    def __init__(self, r, c):
        self.rows = r
        self.cols = c
        self.data = [[0 for i in range(c)] for j in range(r)]

    def __str__(self):
        result = str(self.rows) + " " + str(self.cols)
        for r in range(self.rows):
            result += "\n" + " ".join(str(n) for n in self.data[r])
        return result

    def print(self, cell_width=10):
        print("-" * ((cell_width) * self.cols + 2))
        for r in range(self.rows):
            print("|" + "".join([" " * (cell_width - len("%.6f" % i)) + ("%.6f" % i) for i in self.data[r]]) + "|")
        print("-" * ((cell_width) * self.cols + 2))
        return self

    def map(self, func):
        for r in range(self.rows):
            for c in range(self.cols):
                self.data[r][c] = func(self.data[r][c])
        return self

    def randomize(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.data[r][c] = random.random() * 2 - 1
        return self

    def add(self, val):
        if isinstance(val, (int, float)):
            for r in range(self.rows):
                for c in range(self.cols):
                    self.data[r][c] += val
        elif isinstance(val, Matrix):
            for r in range(self.rows):
                for c in range(self.cols):
                    self.data[r][c] += val.data[r][c]
        else:
            raise ValueError("Matrix operation error")
        return self

    def sub(self, val):
        if isinstance(val, (int, float)):
            for r in range(self.rows):
                for c in range(self.cols):
                    self.data[r][c] -= val
        elif isinstance(val, Matrix):
            for r in range(self.rows):
                for c in range(self.cols):
                    self.data[r][c] -= val.data[r][c]
        else:
            raise ValueError("Matrix operation error")
        return self

    def mult(self, val):
        if isinstance(val, (int, float)):
            for r in range(self.rows):
                for c in range(self.cols):
                    self.data[r][c] *= val
        elif isinstance(val, Matrix):
            for r in range(self.rows):
                for c in range(self.cols):
                    self.data[r][c] *= val.data[r][c]
        else:
            raise ValueError("Matrix operation error")
        return self

    def div(self, val):
        if isinstance(val, (int, float)):
            for r in range(self.rows):
                for c in range(self.cols):
                    self.data[r][c] /= val
        elif isinstance(val, Matrix):
            for r in range(self.rows):
                for c in range(self.cols):
                    self.data[r][c] /= val.data[r][c]
        else:
            raise ValueError("Matrix operation error")
        return self
