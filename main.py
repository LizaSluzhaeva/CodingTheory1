import numpy


def add(x, y):
    """
    Функция вычисляет сумму двух векторов
    """
    return (x + y) % 2


def dot(x, y):
    """
    Функция реализует матричное умножение
    """
    return (x @ y) % 2


def swap_rows(matrix, x, y):
    """
    Функция, меняющая две строки матрицы местами
    """
    if x == y:
        return
    matrix[[x, y]] = matrix[[y, x]]


def sum_rows(matrix, x, y):
    """
    Функция суммирования строк матрицы
    """
    matrix[x] = add(matrix[x], matrix[y])


def REF(matrix):
    # Начинаем обрабатывать нулевую строку - ищем строку у которой первая единица левее чем певая единица других строк
    row = 0
    # Флаг найденной в столбце единицы
    found = False
    # Просмотр столбцов матрицы
    for col in range(matrix.shape[1]):
        i = row
        # Поиск единицы в столбце
        for x in matrix[row:, col]:
            if x:
                # Если единица в столбце уже встречалась, то зануляем ее добавлением текущей строки
                if found:
                    sum_rows(matrix, i, row)
                # Если встретилась первая единица, то меняем строку с обрабатываемой
                else:
                    swap_rows(matrix, row, i)
                    found = True
            i += 1
        if found:
            # Если встретилась единица. то обрабатываем следующую строку
            row += 1
            found = False


def RREF(matrix):
    """
    Функция приводит ступенчатую матрицу к приведенному виду и возвращает ведущие столбцы
    """
    lead = []
    # Поиск ведущих столбцов
    for row in range(matrix.shape[0]):
        col = 0
        # Пропускаем не ведущие столбцы
        while not matrix[row, col]:
            col += 1
        lead.append(col)
        # Перебор элементов ведущего столбца выше обрабатываемой строки
        for i, x in enumerate(matrix[:row, col]):
            # Если встретили единицу, то зануляем ее добавлением обрабатываемой строки
            if x:
                sum_rows(matrix, i, row)
    return lead


def contains(matrix, arr):
    """
    Функция вычисляет входит ли строка в матрицу
    """
    for row in matrix:
        if numpy.all(row == arr):
            return True
    return False


def generate_closure(S):
    """
    Функция вычисляет все кодовые слова путем сложения всех слов из порождающего множества
    """
    closure = [numpy.array(row) for row in S]
    while True:
        new_words = []
        # Складываем все слова друг с другом
        for word1 in closure:
            for word2 in closure:
                word3 = add(word1, word2)
                # Если слово новое, то запоминаем его для дальнейшего использования
                if not contains(closure, word3) and not contains(new_words, word3):
                    new_words.append(word3)
        # Если не встретили новых слов
        if len(new_words) == 0:
            break
        else:
            closure.extend(new_words)
    return closure


def all_words(length):
    """
    Функция генерирует все двоичные слова длины length
    """
    if length == 1:
        yield [0]
        yield [1]
    elif length > 1:
        words = list(all_words(length - 1))
        for word in words:
            yield [0] + word
        for word in words:
            yield [1] + word


class LinearCode:
    def __init__(self, S):
        self.G = numpy.array(S)

        REF(self.G)

        # Удяление нулевых строк
        last_idx = self.G.shape[0] - 1
        i = last_idx
        while not numpy.any(self.G[i]):
            i -= 1
        if i != last_idx:
            self.G = self.G[:i + 1]

        self.k, self.n = self.G.shape

        G_ = self.G.copy()
        lead = RREF(G_)
        not_lead = []
        i = 0
        # Определяем номера неведущих столбцов
        for col in range(self.n):
            if i >= len(lead) or lead[i] != col:
                not_lead.append(col)
            else:
                i += 1
        # Создаем матрицу неведущих столбцов
        X = G_[:, not_lead]
        self.H = numpy.zeros((self.n, self.k), dtype=int)
        i = 0
        # Формируем матрицу H, поместив в строки на позиции ведущих столбцов строки из X,
        # а в остальные - строки из единичной матрицы
        for row in range(self.n):
            if i < len(lead) and row == lead[i]:
                self.H[row] = X[i]
                i += 1
            else:
                self.H[row, row - i] = 1
        # Вычисляем кодовое расстояние
        self.d = self.n
        for i, x in enumerate(self.G):
            for j, y in enumerate(self.G):
                if i != j:
                    d = sum(add(x, y))
                    if d < self.d:
                        self.d = d
        self.t = self.d - 1


def main():
    S = [[1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
         [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
         [1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 1, 1, 1]]

    code = LinearCode(S)

    print('G:')
    print(code.G)
    print(f'n = {code.n}, k = {code.k}')
    print('H:')
    print(code.H)

    u = numpy.array([1, 0, 1, 1, 0])
    v = dot(u, code.G)
    print('v:')
    print(v)
    print('v @ H:')
    print(dot(v, code.H))

    closure1 = generate_closure(S)
    # Умножаем каждое двоичное слово длины k на матрицу G
    closure2 = [dot(numpy.array(row), code.G) for row in all_words(code.k)]
    # Сравниваем полученные множества кодовых слов
    is_equal = len(closure1) == len(closure2) and all(contains(closure2, row) for row in closure1)
    print(f'closure1 is {"equal" if is_equal else "not equal"} to closure2')

    print(f'd = {code.d}, t = {code.t}')

    e1 = numpy.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    print('(v + e1) @ H:')
    print(dot(add(v, e1), code.H))
    e2 = numpy.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    print('(v + e2) @ H:')
    print(dot(add(v, e2), code.H))


if __name__ == '__main__':
    main()
