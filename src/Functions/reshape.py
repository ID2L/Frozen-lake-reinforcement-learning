from nptyping import Float, NDArray, Shape, Int
import numpy as np

def reshape_one(Q_sa: NDArray[Shape['StateDimension, ActionDimension'], Float]) ->NDArray[Shape['StateDimension, ActionDimension'], Int] :
    reshaped_Q_sa = np.zeros(Q_sa.shape, int)
    num_rows, num_cols = Q_sa.shape
    for row_index in range(num_rows):     
        reshaped_Q_sa[row_index, int(np.argmax(Q_sa[row_index,]))] = 1
    return reshaped_Q_sa

def reshape_order(Q_sa: NDArray[Shape['StateDimension, ActionDimension'], Float]) ->NDArray[Shape['StateDimension, ActionDimension'], Int] :
    reshaped_Q_sa = np.full(Q_sa.shape, 0)
    num_rows, num_cols = Q_sa.shape
    for row_index in range(num_rows):     
        reshaped_Q_sa[row_index, ] = np.argsort(Q_sa[row_index,])
    return reshaped_Q_sa


# q = np.random.random((10, 3))
# q_shape = reshape(q)
# print(q)
# print(q_shape)