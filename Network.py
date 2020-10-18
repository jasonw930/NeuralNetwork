import pygame
import math
import random
from numpy import interp
from Matrix import *


class Network:
    def __init__(self, *l):
        self.selected = -1
        self.layers = list(l)
        self.learning_rate = 0.001
        self.values = []
        self.errors = []
        self.weights = []
        self.biases = []
        for i in range(len(self.layers)):
            self.values.append(Matrix(self.layers[i], 1))
            self.errors.append(Matrix(self.layers[i], 1))
        self.biases.append(Matrix(self.layers[0], 1))
        for i in range(len(self.layers) - 1):
            self.weights.append(Matrix(self.layers[i+1], self.layers[i]).randomize())
            self.biases.append(Matrix(self.layers[i+1], 1).randomize())
        # Use ReLU instead of sigmoid
        self.activation = lambda a: 1/(1+math.exp(-a))
        self.d_activation = lambda a: a * (1-a)
        # self.activation = lambda a: max(0, a)
        # self.d_activation = lambda a: 1 if a > 0 else 0

        # Flower (4, 3, 3)
        # self.training_data = [s.split(",") for s in open("iris.data", "r").read().split("\n")]
        # self.training_data = [([float(n) for n in s[:-1]], s[-1]) for s in self.training_data]
        # self.training_data = [(Matrix.from_list(n), Matrix.from_list([
        #     1 if s == "Iris-setosa" else 0,
        #     1 if s == "Iris-versicolor" else 0,
        #     1 if s == "Iris-virginica" else 0
        # ])) for n, s in self.training_data]

        # Mnist(784, 32, 10)
        self.training_data = []
        with open("train-images.idx3-ubyte", "rb") as in_file:
            with open("train-labels.idx1-ubyte", "rb") as out_file:
                self.in_s = in_file.read()
                self.out_s = out_file.read()

    def clear(self):
        for m in self.values:
            m.mult(0)
        return self

    def feedforward(self, value):
        if value is not None:
            self.values[0].mult(0).add(value)
        for i in range(len(self.layers) - 1):
            self.values[i+1].mult(0).add(Matrix.matrix_product(self.weights[i], self.values[i])).add(self.biases[i+1]).map(self.activation)
        return self.values[-1]

    def train(self, values, target):
        # Error: δ = ∂C/∂z = ∂C/∂a * ∂a/∂z
        # C = 1/2 (y-a)^2, ∂C/∂a = (y-a)
        # a = self.activation, ∂a/∂z = self.d_activation
        # Bias Gradient: ∂C/∂b = δ
        # Weight Gradient: ∂C/∂w = (ain)(δout)
        cost = 0
        case_error = []
        for i in range(len(self.layers)):
            self.errors[i].mult(0)
            case_error.append(Matrix(self.layers[i], 1))
        for i in range(len(values)):
            if (i+1) % 64 == 0: print(i+1, end=" ", flush=True)
            self.feedforward(values[i])
            # ** Important Note **
            # σ' is implemented as x*(1-x) instead of σ(x)*(1-σ(x))
            # x*(1-x) can only be applied to a instead of z, since z isn't stored
            # Resulting in ∂a/∂z = a*(1-a) = σ(z)*(1-σ(z)), as required
            case_error[-1].mult(0).add(target[i]).sub(self.values[-1])
            case_error[-1].mult(Matrix.copy(self.values[-1]).map(self.d_activation))
            for j in range(len(self.layers) - 2, 0, -1):
                case_error[j].mult(0).add(Matrix.matrix_product(Matrix.transpose(self.weights[j]), case_error[j+1]))
                case_error[j].mult(Matrix.copy(self.values[j]).map(self.d_activation))
            # cost += 1/2 * sum([n ** 2 for n in Matrix.to_list(Matrix.copy(target[i]).sub(self.values[-1]))])
            for j in range(len(self.layers)):
                self.errors[j].add(case_error[j])
        if len(values) >= 64: print()
        for i in range(len(self.layers)):
            self.errors[i].div(len(values))
        cost /= len(values)

        for i in range(1, len(self.layers)):
            delta_bias = Matrix.copy(self.errors[i]).mult(self.learning_rate)
            self.biases[i].add(delta_bias)
        for i in range(len(self.layers) - 1):
            delta_weight = Matrix.matrix_product(self.errors[i+1], Matrix.transpose(self.values[i])).mult(self.learning_rate)
            self.weights[i].add(delta_weight)

    def test_train(self, value, target):
        self.values[0].mult(0).add(value)
        for i in range(1, len(self.layers)):
            self.values[i].mult(0).add(Matrix.matrix_product(self.weights[i-1], self.values[i-1]))
            self.values[i].add(self.biases[i]).map(self.activation)

        self.errors[-1].mult(0).add(target).sub(self.values[-1])
        for i in range(len(self.layers) - 2, 0, -1):
            self.errors[i].mult(0).add(Matrix.matrix_product(Matrix.transpose(self.weights[i]), self.errors[i+1]))
        for i in range(len(self.layers) - 1, 0, -1):
            gradient = Matrix.copy(self.values[i]).map(self.d_activation)
            gradient.mult(self.errors[i]).mult(self.learning_rate)
            self.biases[i].add(gradient)
            self.weights[i-1].add(Matrix.matrix_product(gradient, Matrix.transpose(self.values[i-1])))

    def gen_training_data(self, i):
        if len(self.training_data) > i: return self.training_data[i]
        in_m = Matrix.from_list([c/256 for c in self.in_s[16+784*i:800+784*i]])
        out_m = Matrix.from_list([0]*10)
        out_m.data[self.out_s[8+i]][0] = 1
        return in_m, out_m

    def auto_train(self, batch_size):
        # self.test_train(*self.generate_training())
        # return
        values, output = [], []
        for n in range(batch_size):
            i, o = self.gen_training_data(random.randint(0, 59999))
            values.append(i)
            output.append(o)
        self.train(values, output)

    def print(self):
        for m in self.weights:
            m.print()
        for m in self.biases:
            print(Matrix.to_list(m))

    def display(self, window, font, width, height, hdist, vdist, rad):
        nodes = []
        for i in range(len(self.layers)):
            nodes.append([{"x": width/2 - (len(self.layers)-1)/2*hdist + i*hdist, "y": height/2 - (self.layers[i]-1)/2*vdist + j*vdist} for j in range(self.layers[i])])
        for i in range(len(self.layers)-1):
            for j in range(self.layers[i]):
                for k in range(self.layers[i+1]):
                    start = (int(nodes[i][j]["x"]), int(nodes[i][j]["y"]))
                    end = (int(nodes[i+1][k]["x"]), int(nodes[i+1][k]["y"]))
                    color = pygame.Color(0)
                    color.hsva = (interp(self.weights[i].data[k][j], [-1.0, 1.0], [0.0, 120.0]), 100, 100, 100)
                    pygame.draw.line(window, color, start, end, 4)
        for i in range(len(self.layers)):
            for j in range(self.layers[i]):
                n = nodes[i][j]
                pygame.draw.circle(window, (255, 255, 255), (int(n["x"]), int(n["y"])), rad)
                if i > 0:
                    color = pygame.Color(0)
                    color.hsva = (interp(self.biases[i].data[j][0], [-1.0, 1.0], [0.0, 120.0]), 100, 100, 100)
                    pygame.draw.circle(window, color, (int(n["x"]), int(n["y"])), rad, 4)
                render = font.render("%.3f" % self.values[i].data[j][0], False, (0, 0, 0))
                window.blit(render, (int(n["x"]) - render.get_size()[0] // 2, int(n["y"]) - render.get_size()[1] // 2))
        if self.selected >= 0:
            select_rect = (nodes[0][self.selected]["x"]-rad-10, nodes[0][self.selected]["y"]-rad-10, 2*rad+20, 2*rad+20)
            pygame.draw.rect(window, (255, 255, 255), select_rect, 4)

    def save(self, name):
        result = " ".join(str(n) for n in self.layers)
        for w in self.weights:
            result += "\n\n" + str(w)
        for b in self.biases:
            result += "\n\n" + str(b)
        open(name, "w").write(result)

    def load(self, name):
        try:
            val = open(name, "r").read().split("\n\n")
            self.layers = [int(s) for s in val[0].split()]
            self.weights = [Matrix.from_str(s) for s in val[1:len(self.layers)]]
            self.biases = [Matrix.from_str(s) for s in val[len(self.layers):]]
            print("Finished Loading", name)
        except Exception:
            pass
