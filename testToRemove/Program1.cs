
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Linq;

class Program
{
    static Matrix<double> InputMatrix()
    {
        // Задання розміру квадратної матриці
        Console.Write("Введіть розмірність квадратної матриці: ");
        int n = int.Parse(Console.ReadLine());
        // Створення матриці A з нулями за розміром n x n
        Matrix<double> A = Matrix<double>.Build.Dense(n, n);
        // Заповнення матриці A введеними значеннями з клавіатури
        for (int i = 0; i < n; i++)
        {
            Console.Write($"Введіть рядок {i + 1} матриці A (через пробіл): ");
            double[] row = Array.ConvertAll(Console.ReadLine().Split(), double.Parse);
            A.SetRow(i, row);
        }
        return A;
    }

    static Vector<double> InputVector(int n)
    {
        // Створення вектора b з нулями за розміром n
        Vector<double> b = Vector<double>.Build.Dense(n);
        // Заповнення вектора b введеними значеннями з клавіатури
        Console.Write("Введіть вектор b (через пробіл): ");
        double[] bArray = Array.ConvertAll(Console.ReadLine().Split(), double.Parse);
        b.SetValues(bArray);
        return b;
    }

    static Matrix<double> CholeskyDecomposition(Matrix<double> A)
    {
        // Присвоюємо n розмір матриці A
        int n = A.RowCount;
        // Створення матриці L з нулями за розміром n x n
        Matrix<double> L = Matrix<double>.Build.Dense(n, n);

        // Обчислення елементів матриці L
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                // Обчислення суми елементів матриці L
                double s = 0;
                for (int k = 0; k < j; k++)
                {
                    s += L[i, k] * L[j, k];
                }

                if (i == j)
                {
                    // Обчислення діагональних елементів матриці L
                    L[i, j] = Math.Sqrt(A[i, i] - s);
                }
                else
                {
                    // Обчислення недіагональних елементів матриці L
                    L[i, j] = (1 / L[j, j]) * (A[i, j] - s);
                }
            }
        }

        return L;
    }

    static Vector<double> ForwardReplace(Matrix<double> L, Vector<double> b)
    {
        // Отримуємо розмір матриці L та вектора b
        int n = L.RowCount;
        // Створюємо вектор y з нулями розміром n
        Vector<double> y = Vector<double>.Build.Dense(n);
        // Обчислення елементів вектора y
        for (int i = 0; i < n; i++)
        {
            // Обчислення i-го елемента вектора y
            y[i] = (b[i] - L.Row(i).SubVector(0, i) * y.SubVector(0, i)) / L[i, i];
        }
        return y;
    }

    static Vector<double> BackwardReplace(Matrix<double> L, Vector<double> y)
    {
        // Отримуємо розмір матриці L та вектора y
        int n = L.RowCount;
        // Створення вектора x з нулями розміром n
        Vector<double> x = Vector<double>.Build.Dense(n);
        // Транспонування матриці L
        Matrix<double> LT = L.Transpose();
        // Обчислення елементів вектора x
        for (int i = n - 1; i >= 0; i--)
        {
            // Обчислення i-й елемент вектора x
            x[i] = (y[i] - LT.Row(i).SubVector(i + 1, n - i - 1) * x.SubVector(i + 1, n - i - 1)) / LT[i, i];
        }
        return x;
    }

    static double Determinant(Matrix<double> L)
    {
        return Math.Pow(L.Diagonal().Aggregate(1.0, (acc, val) => acc * val), 2);
    }

    static void Main(string[] args)
    {
        // Введення матриці A та вектора b
        Matrix<double> A = InputMatrix();
        int n = A.RowCount;
        Vector<double> b = InputVector(n);

        // Перевірка на коректність розмірності матриці A та вектора b
        if (A.RowCount != b.Count)
        {
            Console.WriteLine("Розмірність матриці A та вектора b не збігається.");
            return;
        }
        // Перевірка на симетричність та додатню визначеність матриці A
        if (!A.Equals(A.Transpose()) || !A.Evd().EigenValues.All(z => z.Magnitude > 0))
        {
            Console.WriteLine("Матриця A не є симетричною та додатньо визначеною. Розкладу Холецького не існує.");
            return;
        }

        // Виклик розкладу Холецького для матриці A
        Matrix<double> L = CholeskyDecomposition(A);
        // Виклик методу прямої підстановки для y
        Vector<double> y = ForwardReplace(L, b);
        // Виклик зворотньої підстановки для x
        Vector<double> x = BackwardReplace(L, y);
        // Виклик функції знаходження визначника матриці A
        double det = Determinant(L);

        Console.WriteLine("Матриця L:");
        // Покрокове виведення елементів матриці L
        for (int i = 0; i < L.RowCount; i++)
        {
            for (int j = 0; j < L.ColumnCount; j++)
            {
                Console.Write($"{L[i, j]:F4} ");
            }
            Console.WriteLine();
        }
        Console.WriteLine("Розв'язок системи рівнянь:");
        for (int i = 0; i < x.Count; i++)
        {
            Console.WriteLine($"{x[i]:F4}");
        }
        Console.WriteLine("Визначник матриці A:");
        Console.WriteLine($"{det:F4}");
        Console.ReadLine();
    }
}