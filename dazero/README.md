## Math

$$
\mathbf{Y} = \mathbf{W} \mathbf{X}
$$

若 $\mathbf{Y}$ 经过 *某种运算* 得到了标量 $\mathbf{L}$, 则标量 $\mathbf{L}$ 对矩阵 $\mathbf{X}$ 求偏导

$$
\begin{aligned}
\frac{\partial \mathbf{L}}{\partial \mathbf{X}} &= \frac{\partial \mathbf{L}}{\partial \mathbf{Y}} ~ \mathbf{W} ^ T
\\
\frac{\partial \mathbf{L}}{\partial \mathbf{W}} &= \mathbf{X} ^ T ~ \frac{\partial \mathbf{L}}{\partial \mathbf{Y}}
\end{aligned}
$$
