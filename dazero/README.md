# Math

## Linear Layer

$$
Y = W X
$$

若 $Y$ 经过 *某种运算* 得到了标量 $L$, 则标量 $L$ 对矩阵 $X$ 求偏导

$$
\begin{aligned}
\frac{\partial L}{\partial X} &= \frac{\partial L}{\partial Y} ~ W ^ T
\\
\frac{\partial L}{\partial W} &= X ^ T ~ \frac{\partial L}{\partial Y}
\end{aligned}
$$

## RNN Layer

$$
h_t = \tanh(h_{t-1} W_h + x_t W_x + b)
$$

### LSTM

$$
\begin{aligned}

f_t &= \sigma(x_t W_x^{(f)} + h_{t-1} W_h ^{(f)} + b^{(f)})
\\
i_t &= \sigma(x_t W_x^{(i)} + h_{t-1} W_h ^{(i)} + b^{(i)})
\\
o_t &= \sigma(x_t W_x^{(o)} + h_{t-1} W_h ^{(o)} + b^{(o)})
\\
u_t &= \sigma(x_t W_x^{(u)} + h_{t-1} W_h ^{(u)} + b^{(u)})

\end{aligned}
$$

$$
\begin{aligned}

c_t &= f_t \odot c_{t-1} + i_t \odot u_t
\\
h_t &= o_t \odot \tanh(c_t)

\end{aligned}
$$
