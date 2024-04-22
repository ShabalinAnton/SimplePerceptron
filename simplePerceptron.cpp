
#include <cstddef>
#include <iostream>
#include <locale>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <sys/types.h>
#include <vector>
#include <list>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <limits>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/freeglut.h>

const int window_width = 800;
const int window_height = 600;

int mousePosX = 0;
int mousePosY = 0;

// ------------------------------------------------------------------------------------------------
// Matrix operations
class Matrix
{
    public:
        Matrix(int rows, int cols) : columns(cols), rows(rows) 
        {
            assert(columns * rows > 0);
            matrix_data.resize(columns * rows, 0.0);
        }
        Matrix(const std::vector<std::vector<double>> &matrix);
        Matrix() = delete;

        friend Matrix operator*(const Matrix &lhs, const Matrix &rhs);
        friend Matrix operator+(const Matrix &lhs, const Matrix &rhs);
        friend Matrix operator-(const Matrix &lhs, const Matrix &rhs);

        friend Matrix operator*(const Matrix &lhs, double scalar);
        friend Matrix operator+(const Matrix &lhs, double scalar);
        friend Matrix operator-(const Matrix &lhs, double scalar);
        friend Matrix operator-(double scalar, const Matrix &lhs);
        
        double& operator()(int r, int c) {return value(r, c);}
        const double operator()(int r, int c) const {return value(r, c);}

        double& value(int r, int c)
        {
            int index = r * columns + c;
            assert(index < matrix_data.size());
            return matrix_data[index];
        }

        const double value(int r, int c) const
        {
            int index = r * columns + c;
            assert(index < matrix_data.size());
            return matrix_data[index];
        }

        int getColumns() const { return columns; }
        int getRows() const { return rows; }

        std::vector<double>& getData() { return matrix_data; }

        // Return transposed matrix
        Matrix transposed() const;

    private:
        int columns = 0;
        int rows = 0;
        std::vector<double> matrix_data;
};
// ------------------------------------------------------------------------------------------------
Matrix::Matrix(const std::vector<std::vector<double>> &matrix)
{
    assert(matrix.size() > 0);

    rows = (int)matrix.size();
    columns = (int)matrix[0].size();

    assert(columns * rows > 0);
    matrix_data.resize(columns * rows, 0.0);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < columns; c++) {
            value(r, c) = matrix[r][c];
        }
    }
} 
// ------------------------------------------------------------------------------------------------
Matrix operator*(const Matrix &lhs, const Matrix &rhs)
{
    assert(lhs.getColumns() == rhs.getRows());
    assert(lhs.matrix_data.size() != 0);
    assert(rhs.matrix_data.size() != 0);

    Matrix result(lhs.rows, rhs.columns);

    int count = lhs.columns;

    for (int r = 0; r < result.rows; r++) {
        for (int c = 0; c < result.columns; c++) {
            result(r, c) = 0;
            for (int k = 0; k < count; k++) {
                result(r,c) += lhs(r, k) * rhs(k, c);
            }
        }
    }
    return result;
}
// ------------------------------------------------------------------------------------------------
Matrix operator*(const Matrix&lhs, double scalar)
{
    Matrix result(lhs);

    for (double &v : result.matrix_data) {
        v *= scalar;
    }
    return result;
}
// ------------------------------------------------------------------------------------------------
Matrix operator+(const Matrix &lhs, const Matrix &rhs)
{
    assert(lhs.matrix_data.size() == rhs.matrix_data.size());
    assert(lhs.matrix_data.size() != 0);

    Matrix result(lhs);
    for (int i = 0; i < (int)rhs.matrix_data.size(); i++) {
        result.matrix_data[i] += rhs.matrix_data[i];
    }
    return result;
}
// ------------------------------------------------------------------------------------------------
Matrix operator-(const Matrix &lhs, const Matrix &rhs)
{
    assert(lhs.matrix_data.size() == rhs.matrix_data.size());
    assert(lhs.matrix_data.size() != 0);

    Matrix result(lhs);
    for (int i = 0; i < (int)rhs.matrix_data.size(); i++) {
        result.matrix_data[i] -= rhs.matrix_data[i];
    }
    return result;
}
// ------------------------------------------------------------------------------------------------
Matrix operator+(const Matrix&lhs, double scalar)
{
    Matrix result(lhs);

    for (double &v : result.matrix_data) {
        v += scalar;
    }
    return result;
}
// ------------------------------------------------------------------------------------------------
Matrix operator-(const Matrix&lhs, double scalar)
{
    Matrix result(lhs);

    for (double &v : result.matrix_data) {
        v -= scalar;
    }
    return result;
}
// ------------------------------------------------------------------------------------------------
Matrix operator-(double scalar, const Matrix&lhs)
{
    Matrix result(lhs);
    for (double &v : result.matrix_data) {
        v = scalar - v;
    }
    return result;
}
// ------------------------------------------------------------------------------------------------
Matrix Matrix::transposed() const
{
    Matrix result(columns, rows);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < columns; c++) {
            result(c, r) = value(r, c);
        }
    }
    return result;
}
// ------------------------------------------------------------------------------------------------
// Matrix functions and NN operations

#define RANDOMIZE_W(w) \
    for (double &v : w.getData()) { \
        v = (double)std::rand() / (double)RAND_MAX; \
    }

// Multiply matrices element-by-element
auto multiplyElements = [](const Matrix& m1, const Matrix& m2) 
{
    assert(m1.getColumns() == m2.getColumns());
    assert(m1.getRows() == m2.getRows());

    Matrix result(m1);
    for (size_t r = 0; r < m1.getRows(); r++) {
        for (size_t c = 0; c < m1.getColumns(); c++) {
            result(r, c) = m1(r, c) * m2(r, c);
        }
    }
    return result;
};

// Activation functions
auto sigmoid = [](const Matrix& m) -> Matrix
{
    Matrix result(m);
    for (double &v : result.getData()) {
        v = 1.0 / (1.0 + std::exp(-v));
    }
    return result;
};

// Derivatives
auto d_sigmoid = [](const Matrix& m) -> Matrix
{
    Matrix result(m);
    for (int r = 0; r < m.getRows(); r++) {
        double &v = result(r, 0);
        v = v * (1.0 - v);
    }
    return result;
};

// ------------------------------------------------------------------------------------------------
// Logs
void print_matrix(const Matrix& matrix)
{
    for(int r = 0; r < matrix.getRows(); r++) {
        for (int c = 0; c < matrix.getColumns(); c++) {
            std::cout << matrix(r, c) << "\t";
        }
        std::cout << std::endl;
    }
}

void print_matrix(const std::string &name, const Matrix& matrix)
{
    std::cout << name << ":" << std::endl;
    print_matrix(matrix);
}

#define PRINT_MATRIX(x) print_matrix(#x, x);

// ------------------------------------------------------------------------------------------------
// Simple neural network class (perceptron)
class NeuralNetwork
{
    public:
        NeuralNetwork(std::vector<int> topology)
            : _topology(topology)
        {
            assert(topology.size() > 2);
            
            // Initializing network weights and biases
            for (size_t t = 0; t < topology.size() - 1; t++) {

                int t0 = topology[t];
                int t1 = topology[t + 1];

                Matrix W(t1, t0);
                Matrix B(t1, 1);

                // Applying random values to weights and biases
                RANDOMIZE_W(W);
                RANDOMIZE_W(B);

                _weights.push_back(W);
                _biases.push_back(B);
                _values.push_back(Matrix(t0, 1));
            }
            _values.push_back(Matrix(topology.back(), 1));
        }

        NeuralNetwork() = delete;

        // training calculations
        int train(const std::vector<Matrix>& inputs, const std::vector<Matrix>& outputs, double learning_rate = 0.1, unsigned int epoches_count = 10000)
        {
            assert(!inputs.empty() && !outputs.empty());
            assert(inputs.size() == outputs.size());

            _learn = learning_rate;

            for (unsigned int epoch = 0; epoch < epoches_count; epoch++) {

                double epoch_error = 0.0;

                for (size_t t = 0; t < inputs.size(); t++) {
                    _values[0] = inputs[t];
                    const Matrix& output = outputs[t];

                    forward_step();
                    back_propagation(output);

                    Matrix error = output - _values.back();
                    double input_errors = 0.0;
                    for (int r = 0; r < error.getRows(); r++) {
                        input_errors += pow(error(r, 0), 2.0);
                    }
                    input_errors /= error.getRows();
                    epoch_error += input_errors;
                }
                epoch_error /= inputs.size();

                std::cout << "Epoch " << epoch << " RSS: " << epoch_error << std::endl;
            } // for (unsigned int epoch = 0; epoch < epoches_count; epoch++)

            return 0;
        }

        // calculate solution
        void calculate(const Matrix& inputs, Matrix &outputs)
        {
            _values[0] = inputs;
            forward_step();
            outputs = _values.back();
        }

    private: 
        // forward propagation step - calculating nodes values
        void forward_step()
        {
            for (size_t t = 0; t < _topology.size() - 1; t++) {
                _values[t + 1] = sigmoid(_weights[t] * _values[t] + _biases[t]);
            }
        }

        // back propagation - calculating errors
        void back_propagation(const Matrix& target_output)
        {
            // Errors matrix - for all layers
            std::vector<Matrix> errors; 

            // calculate last error: Error = Answer - Last output
            errors.push_back(target_output - _values.back());

            // back propagating the error from output layer to input layer
            // updating weights and biases
            for (int e = (int)_topology.size() - 2; e >= 0; e--) {

                // calculate errors for previous layers
                Matrix E = errors.back();

                errors.push_back(_weights[e].transposed() * E);
                
                // Correcting weights
                // W'i1 = Wi1 + learn_coeff * <E_i1 * (dY_11(e) / de)> * X1^t => 
                // W'i1 = Wi1 + learn_coeff * <E_i1 * Y_1 * (1 - Y_1)> * X1^t // for sigmoid func

                // calculating derivative (gradient)
                Matrix grad = multiplyElements(d_sigmoid(_values[e + 1]), E) * _learn;
                Matrix dW = grad * _values[e].transposed();
                
                _biases[e] = _biases[e] + grad;
                _weights[e] = _weights[e] + dW;
            } // for (size_t e = _topology.size() - 1; e >= 0; e--)
        }


    private:
        double _learn = 0.1;
        std::vector<Matrix> _weights;
        std::vector<Matrix> _biases;
        std::vector<Matrix> _values;
        std::vector<int>    _topology;
};
// ------------------------------------------------------------------------------------------------
// Point structure.
struct Point
{
    double x = 0.0, y = 0.0;
    double pointTypeScore = 0.0;
};

const double bluePoint = 0.01;
const double redPoint = 0.99;

std::list<Point> input_data;    // An array of input data;
std::list<Point> solution_data;

// ------------------------------------------------------------------------------------------------
void train_network()
{
    if (input_data.empty()) {
        return;
    }

    solution_data.clear();

// Neural network structure
/*
Input layer:      1-st Layer:    2-nd layer:    3-rd layer (output):
            (w_i)            (w_1)          (w_2)
        e_i              e_1            (e_2)       (e)
                    [Y11]       
    [X1] ------                    [Y21]
                    [Y12]                          [Y31] --->  Y
    [X2] ------                    [Y22]
                    [Y13]

    [B]             [B1]           [B2]
   (bias)

   w - weights, e - errors
   X - input neurons, Y - output neuron, Yxx - hidden neurons, B - bias neurons
*/
    std::vector<int> topology = {2, 3, 2, 1};

    std::vector<Matrix> inputs;
    std::vector<Matrix> outputs;

    for (const Point& p : input_data) {
        inputs.push_back( Matrix{{{p.x / (double)window_width}, {p.y / (double)window_height}}} );
        outputs.push_back( Matrix{{{p.pointTypeScore}}} );
    }

    NeuralNetwork NN(topology);
    NN.train(inputs, outputs, 0.01, 1000000);

    std::cout << "Train is done." << std::endl;

    for (int w = 0; w < window_width; w++) {
        for (int h = 0; h < window_height; h++) {

            double x = (double)w;
            double y = (double)h;

            Matrix X(2,1);
            Matrix Y(1,1);
            
            X(0, 0) = x / (double)window_width;
            X(1, 0) = y / (double)window_height;

            NN.calculate(X, Y);

            double solution = Y(0, 0);
            solution_data.push_back({x, y, solution});
        }
    }
}

// ------------------------------------------------------------------------------------------------
void init() 
{
    glClearColor(1.f, 1.f, 1.f, 0.f);
	glViewport(0,0,window_width,window_height);  
    glMatrixMode(GL_PROJECTION); 
	glLoadIdentity();  
    gluOrtho2D(0.0,(GLdouble)window_width,0.0,(GLdouble)window_height);  
    glMatrixMode(GL_MODELVIEW);
    glPointSize(4.0f);
}

bool isRealEqual(double a, double b)
{
    using namespace std;
    return abs(a - b) <= numeric_limits<double>::epsilon() * max(abs(a), abs(b));
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT);

    glBegin(GL_POINTS);

    for (auto p: solution_data) {

        double blueColor = 1.0 - p.pointTypeScore;
        double redColor = p.pointTypeScore;

        glColor4d(redColor, 0.4, blueColor, 0.9);
        glVertex2f(p.x, glutGet(GLUT_WINDOW_HEIGHT) - p.y);
    }

    for(auto p : input_data) {
        if (isRealEqual(p.pointTypeScore, bluePoint)) {
            glColor3f(0.f, 0.f, 1.f);
        } else if (isRealEqual(p.pointTypeScore, redPoint)) {
            glColor3f(1.f, 0.f, 0.f);
        }
        glVertex2f(p.x, glutGet(GLUT_WINDOW_HEIGHT) - p.y);
    }

    glEnd();

	glFlush();
    glutSwapBuffers();
}

void mouseMove(int x, int y)
{
    mousePosX = x;
    mousePosY = y;
}

void mouse(int btn, int state, int x, int y) 
{
    if (state == GLUT_DOWN) {
        input_data.push_back({(double)x, (double)y, (btn == GLUT_LEFT_BUTTON) ? bluePoint : redPoint});
        glutPostRedisplay();
    }
}

void keyboard(unsigned char key, int x, int y)
{
    // Hit "Delete" or "Backspace" key
    if ((int)key == 127 || (int)key == 8) {
        auto itr = std::find_if(input_data.begin(), input_data.end(), [x, y](const Point& p) {
            return (fabs(p.x - (double)x) < 5 ) && (fabs(p.y - (double)y) < 5);
        });
        if (itr != input_data.end()) {
            input_data.erase(itr);
            glutPostRedisplay();
        }
    }

    // space 
    if ((int)key == 32) {
        train_network();
        glutPostRedisplay();
    }

    // enter - clear
    if ((int)key == 13) {
        input_data.clear();
        solution_data.clear();
        glutPostRedisplay();
    }
}

int main(int argc, char** argv)
{
    std::srand(std::time(nullptr));
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Simple Perceptron Test");
    glutDisplayFunc(display);
    init();
    glutMouseFunc(mouse);
    glutPassiveMotionFunc(mouseMove);
    glutKeyboardFunc(keyboard);
    glutMainLoop();

    return EXIT_SUCCESS;
}
