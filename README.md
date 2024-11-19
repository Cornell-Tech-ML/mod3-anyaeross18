# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


## 3.1 & 3.2 Parallel Analytics Script <br>
## MAP <br>

================================================================================ <br>
Parallel Accelerator Optimizing: Function tensor_map.<locals>._map, <br>
C:\Users\15184\OneDrive\Documents\Machine Learning <br>
Engineering\mod3-anyaeross18\minitorch\fast_ops.py (174) <br>
================================================================================ <br>

Parallel loop listing for Function tensor_map.<locals>._map, <br>
C:\Users\15184\OneDrive\Documents\Machine Learning Engineering\mod3-anyaeross18\minitorch\fast_ops.py (174) <br>
-----------------------------------------------------------------------------|loop #ID <br>
    def _map(                                                                | <br>
        out: Storage,                                                        | <br>
        out_shape: Shape,                                                    | <br>
        out_strides: Strides,                                                | <br>
        in_storage: Storage,                                                 | <br>
        in_shape: Shape,                                                     | <br>
        in_strides: Strides,                                                 | <br>
    ) -> None:                                                               | <br>
        if len(out_strides) != len(in_strides):                              | <br>
            strides_match = False                                            | <br>
        elif (out_strides != in_strides).any():------------------------------| #0 <br>
            strides_match = False                                            | <br>
        elif (out_shape != in_shape).any():----------------------------------| #1 <br>
            strides_match = False                                            | <br>
        else:                                                                | <br>
            strides_match = True                                             | <br>
                                                                             | <br>
        if strides_match:                                                    | <br>
            for i in prange(out.size):---------------------------------------| #2 <br>
                out[i] = fn(in_storage[i])                                   | <br>
        else:                                                                | <br>
            for i in prange(out.size):---------------------------------------| #3 <br>
                out_index = np.zeros_like(out_shape, dtype=np.int32)         | <br>
                in_index = np.zeros_like(in_shape, dtype=np.int32)           | <br>
                to_index(i, out_shape, out_index)                            | <br>
                broadcast_index(out_index, out_shape, in_shape, in_index)    | <br>
                position_in = index_to_position(in_index, in_strides)        | <br>
                position_out = index_to_position(out_index, out_strides)     | <br>
                out[position_out] = fn(in_storage[position_in])              | <br>
--------------------------------- Fusing loops --------------------------------- <br>
Attempting fusion of parallel loops (combines loops with similar properties)... <br>
Following the attempted fusion of parallel for-loops there are 4 parallel for- <br>
loop(s) (originating from loops labelled: #0, #1, #2, #3). <br>
-------------------------------------------------------------------------------- <br>
----------------------------- Before Optimisation ------------------------------ <br>
-------------------------------------------------------------------------------- <br>
------------------------------ After Optimisation ------------------------------ <br>
Parallel structure is already optimal. <br>
-------------------------------------------------------------------------------- <br>
-------------------------------------------------------------------------------- <br>

---------------------------Loop invariant code motion--------------------------- <br>
Allocation hoisting: <br>
No allocation hoisting found <br>
None <br>

## ZIP<br>

================================================================================<br>
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,<br>
C:\Users\15184\OneDrive\Documents\Machine Learning<br>
Engineering\mod3-anyaeross18\minitorch\fast_ops.py (230)<br>
================================================================================<br><br>

Parallel loop listing for  Function tensor_zip.<locals>._zip, C:\Users\15184\OneDrive\Documents\Machine Learning Engineering\mod3-anyaeross18\minitorch\fast_ops.py (230)<br>
----------------------------------------------------------------------------------------|loop #ID<br>
    def _zip(                                                                           |<br>
        out: Storage,                                                                   |<br>
        out_shape: Shape,                                                               |<br>
        out_strides: Strides,                                                           |<br>
        a_storage: Storage,                                                             |<br>
        a_shape: Shape,                                                                 |<br>
        a_strides: Strides,                                                             |<br>
        b_storage: Storage,                                                             |<br>
        b_shape: Shape,                                                                 |<br>
        b_strides: Strides,                                                             |<br>
    ) -> None:                                                                          |<br>
        if len(out_strides) != len(a_strides):                                          |<br>
            strides_match = False                                                       |<br>
        elif len(out_strides) != len(b_strides):                                        |<br>
            strides_match = False                                                       |<br>
        elif (out_strides != a_strides).any():------------------------------------------| #4<br>
            strides_match = False                                                       |<br>
        elif (out_strides != b_strides).any():------------------------------------------| #5<br>
            strides_match = False                                                       |<br>
        elif (out_shape != a_shape).any():----------------------------------------------| #6<br>
            strides_match = False                                                       |<br>
        elif (out_shape != b_shape).any():----------------------------------------------| #7<br>
            strides_match = False                                                       |<br>
        else:                                                                           |<br>
            strides_match = True                                                        |<br>
                                                                                        |<br>
        # If strides match, apply the function directly                                 |<br>
        if strides_match:                                                               |<br>
            for i in prange(out.size):--------------------------------------------------| #8<br>
                out[i] = fn(a_storage[i], b_storage[i])                                 |<br>
        else:                                                                           |<br>
            for i in prange(out.size):--------------------------------------------------| #9<br>
                out_index = np.zeros_like(out_shape, dtype=np.int32)                    |<br>
                a_index = np.zeros_like(a_shape, dtype=np.int32)                        |<br>
                b_index = np.zeros_like(b_shape, dtype=np.int32)                        |<br>
                                                                                        |<br>
                to_index(i, out_shape, out_index)                                       |<br>
                position_out = index_to_position(out_index, out_strides)                |<br>
                broadcast_index(out_index, out_shape, a_shape, a_index)                 |<br>
                position_a = index_to_position(a_index, a_strides)                      |<br>
                broadcast_index(out_index, out_shape, b_shape, b_index)                 |<br>
                position_b = index_to_position(b_index, b_strides)                      |<br>
                out[position_out] = fn(a_storage[position_a], b_storage[position_b])    |<br>
--------------------------------- Fusing loops ---------------------------------<br>
Attempting fusion of parallel loops (combines loops with similar properties)...<br>
Following the attempted fusion of parallel for-loops there are 6 parallel for-<br>
loop(s) (originating from loops labelled: #4, #5, #6, #7, #8, #9).<br>
--------------------------------------------------------------------------------<br>
----------------------------- Before Optimisation ------------------------------<br>
--------------------------------------------------------------------------------<br>
------------------------------ After Optimisation ------------------------------<br>
Parallel structure is already optimal.<br>
--------------------------------------------------------------------------------<br>
--------------------------------------------------------------------------------<br><br>

---------------------------Loop invariant code motion---------------------------<br>
Allocation hoisting:<br>
No allocation hoisting found<br>
None<br>

## REDUCE<br>
<br>
================================================================================<br>
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,<br>
C:\Users\15184\OneDrive\Documents\Machine Learning<br>
Engineering\mod3-anyaeross18\minitorch\fast_ops.py (298)<br>
================================================================================<br>
<br>
<br>
Parallel loop listing for  Function tensor_reduce.<locals>._reduce, C:\Users\15184\OneDrive\Documents\Machine Learning Engineering\mod3-anyaeross18\minitorch\fast_ops.py (298)<br>
----------------------------------------------------------------------------|loop #ID<br>
    def _reduce(                                                            |<br>
        out: Storage,                                                       |<br>
        out_shape: Shape,                                                   |<br>
        out_strides: Strides,                                               |<br>
        a_storage: Storage,                                                 |<br>
        a_shape: Shape,                                                     |<br>
        a_strides: Strides,                                                 |<br>
        reduce_dim: int,                                                    |<br>
    ) -> None:                                                              |<br>
        # reduce_size = a_shape[reduce_dim]                                 |<br>
        for i in prange(out.size):------------------------------------------| #10<br>
            out_index = np.zeros_like(out_shape, dtype=np.int32)            |<br>
            # reduce_size = a_shape[reduce_dim]                             |<br>
            to_index(i, out_shape, out_index)                               |<br>
            o = index_to_position(out_index, out_strides)                   |<br>
            position_a_base = index_to_position(out_index, a_strides)       |<br>
            temp = a_storage[position_a_base]                               |<br>
            for j in range(1, a_shape[reduce_dim]):                         |<br>
                position_a = position_a_base + j * a_strides[reduce_dim]    |<br>
                temp = fn(temp, float(a_storage[position_a]))               |<br>
            out[o] = temp                                                   |<br>
--------------------------------- Fusing loops ---------------------------------<br>
Attempting fusion of parallel loops (combines loops with similar properties)...<br>
Following the attempted fusion of parallel for-loops there are 1 parallel for-<br>
loop(s) (originating from loops labelled: #10).<br>
--------------------------------------------------------------------------------<br>
----------------------------- Before Optimisation ------------------------------<br>
--------------------------------------------------------------------------------<br>
------------------------------ After Optimisation ------------------------------<br>
Parallel structure is already optimal.<br>
--------------------------------------------------------------------------------<br>
--------------------------------------------------------------------------------<br>
<br>
---------------------------Loop invariant code motion---------------------------<br>
Allocation hoisting:<br>
No allocation hoisting found<br>
None<br>
<br>
<br>
## MATRIX MULTIPLY<br>
<br>
================================================================================<br>
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,<br>
C:\Users\15184\OneDrive\Documents\Machine Learning<br>
Engineering\mod3-anyaeross18\minitorch\fast_ops.py (323)<br>
================================================================================<br>
<br>
<br>
Parallel loop listing for  Function _tensor_matrix_multiply, C:\Users\15184\OneDrive\Documents\Machine Learning Engineering\mod3-anyaeross18\minitorch\fast_ops.py (323)<br>
-------------------------------------------------------------------------------------|loop #ID<br>
def _tensor_matrix_multiply(                                                         |<br>
    out: Storage,                                                                    |<br>
    out_shape: Shape,                                                                |<br>
    out_strides: Strides,                                                            |<br>
    a_storage: Storage,                                                              |<br>
    a_shape: Shape,                                                                  |<br>
    a_strides: Strides,                                                              |<br>
    b_storage: Storage,                                                              |<br>
    b_shape: Shape,                                                                  |<br>
    b_strides: Strides,                                                              |<br>
) -> None:                                                                           |<br>
    """NUMBA tensor matrix multiply function.                                        |<br>
                                                                                     |<br>
    Should work for any tensor shapes that broadcast as long as                      |<br>
                                                                                     |<br>
    ```                                                                              |<br>
    assert a_shape[-1] == b_shape[-2]                                                |<br>
    ```                                                                              |<br>
                                                                                     |<br>
    Optimizations:                                                                   |<br>
                                                                                     |<br>
    * Outer loop in parallel                                                         |<br>
    * No index buffers or function calls                                             |<br>
    * Inner loop should have no global writes, 1 multiply.                           |<br>
                                                                                     |<br>
                                                                                     |<br>
    Args:                                                                            |<br>
    ----                                                                             |<br>
        out (Storage): storage for `out` tensor                                      |<br>
        out_shape (Shape): shape for `out` tensor                                    |<br>
        out_strides (Strides): strides for `out` tensor                              |<br>
        a_storage (Storage): storage for `a` tensor                                  |<br>
        a_shape (Shape): shape for `a` tensor                                        |<br>
        a_strides (Strides): strides for `a` tensor                                  |<br>
        b_storage (Storage): storage for `b` tensor                                  |<br>
        b_shape (Shape): shape for `b` tensor                                        |<br>
        b_strides (Strides): strides for `b` tensor                                  |<br>
                                                                                     |<br>
    Returns:                                                                         |<br>
    -------                                                                          |<br>
        None : Fills in `out`                                                        |<br>
                                                                                     |<br>
    """                                                                              |<br>
    assert a_shape[-1] == b_shape[-2]                                                |<br>
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                           |<br>
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                           |<br>
                                                                                     |<br>
    for i in prange(out.size):-------------------------------------------------------| #11<br>
        out0 = i // (out_shape[-1] * out_shape[-2])                                  |<br>
        out1 = (i % (out_shape[-1] * out_shape[-2])) // out_shape[-1]                |<br>
        out2 = i % out_shape[-1]                                                     |<br>
                                                                                     |<br>
        out_index = (                                                                |<br>
            out0 * out_strides[0] + out1 * out_strides[1] + out2 * out_strides[2]    |<br>
        )                                                                            |<br>
        a_start = out0 * a_batch_stride + out1 * a_strides[-2]                       |<br>
        b_start = out0 * b_batch_stride + out2 * b_strides[-1]                       |<br>
                                                                                     |<br>
        temp = 0.0                                                                   |<br>
        for j in range(a_shape[-1]):                                                 |<br>
            a_index = a_start + j * a_strides[-1]                                    |<br>
            b_index = b_start + j * b_strides[-2]                                    |<br>
            temp += a_storage[a_index] * b_storage[b_index]                          |<br>
        out[out_index] = temp                                                        |<br>
--------------------------------- Fusing loops ---------------------------------<br>
Attempting fusion of parallel loops (combines loops with similar properties)...<br>
Following the attempted fusion of parallel for-loops there are 1 parallel for-<br>
loop(s) (originating from loops labelled: #11).<br>
--------------------------------------------------------------------------------<br>
----------------------------- Before Optimisation ------------------------------<br>
--------------------------------------------------------------------------------<br>
------------------------------ After Optimisation ------------------------------<br>
Parallel structure is already optimal.<br>
--------------------------------------------------------------------------------<br>
--------------------------------------------------------------------------------<br>
<br>
---------------------------Loop invariant code motion---------------------------<br>
Allocation hoisting:<br>
No allocation hoisting found<br>
None<br>
## Timing Graph 3.4:<br>
NumbaPerformanceWarning: Host array used in CUDA kernel will incur copy overhead to/from device.<br>
  warn(NumbaPerformanceWarning(msg))<br>
Running size 64<br>
 Grid size 8 will likely result in GPU under-utilization due to low occupancy.<br>
  warn(NumbaPerformanceWarning(msg))<br>
{'fast': np.float64(0.003506263097127279), 'gpu': np.float64(0.011742512385050455)}<br>
Running size 128<br>
NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.<br>
  warn(NumbaPerformanceWarning(msg))<br>
{'fast': np.float64(0.01695839564005534), 'gpu': np.float64(0.024652481079101562)}<br>
Running size 256<br>
{'fast': np.float64(0.07248171170552571), 'gpu': np.float64(0.11461536089579265)}<br>
Running size 512<br>
{'fast': np.float64(0.3712015946706136), 'gpu': np.float64(0.3677803675333659)}<br>
Running size 1024<br>
{'fast': np.float64(3.6771787802378335), 'gpu': np.float64(1.4124451478322346)}<br>

# Timing summary:<br>
Size: 64<br>
    fast: 0.00351<br>
    gpu: 0.01174<br>
Size: 128<br>
    fast: 0.01696<br>
    gpu: 0.02465<br>
Size: 256<br>
    fast: 0.07248<br>
    gpu: 0.11462<br>
Size: 512<br>
    fast: 0.37120<br>
    gpu: 0.36778<br>
Size: 1024<br>
    fast: 3.67718<br>
    gpu: 1.41245<br>

## 3.5: Training Loss, Accuracy, and Timing<br>
# Small Models:<br>
backend: cpu, dataset: simple, hidden: 100, rate: 0.05, points: 50 <br>
Epoch 0, loss 5.52817496286788, correct 27, time 30.40 seconds
Epoch 10, loss 2.705329348675475, correct 47, time 0.12 seconds
Epoch 20, loss 1.4890510000659218, correct 48, time 0.11 seconds
Epoch 30, loss 0.8635561816490419, correct 49, time 0.13 seconds
Epoch 40, loss 2.102966128787591, correct 50, time 0.11 seconds
Epoch 50, loss 0.3520923121836196, correct 50, time 0.11 seconds
Epoch 60, loss 0.9942342939645927, correct 49, time 0.12 seconds
Epoch 70, loss 0.5556668354600687, correct 49, time 0.12 seconds
Epoch 80, loss 0.8704323319032629, correct 49, time 0.15 seconds
Epoch 90, loss 0.41930459163684614, correct 49, time 0.15 seconds
Epoch 100, loss 0.38002522357595137, correct 50, time 0.12 seconds
Epoch 110, loss 0.22176702976972645, correct 50, time 0.11 seconds
Epoch 120, loss 0.15380724550617056, correct 50, time 0.12 seconds
Epoch 130, loss 0.9830008389780323, correct 49, time 0.14 seconds
Epoch 140, loss 0.8130525341225274, correct 50, time 0.12 seconds
Epoch 150, loss 0.9425638691990154, correct 50, time 0.11 seconds
Epoch 160, loss 0.03902290282321021, correct 50, time 0.14 seconds
Epoch 170, loss 0.5649559134783091, correct 50, time 0.11 seconds
Epoch 180, loss 0.11346897367239869, correct 50, time 0.11 seconds
Epoch 190, loss 0.009645903830819764, correct 50, time 0.13 seconds
Epoch 200, loss 0.7143023831890992, correct 50, time 0.15 seconds
Epoch 210, loss 0.030674413706201026, correct 50, time 0.11 seconds
Epoch 220, loss 0.01696839361679683, correct 50, time 0.16 seconds
Epoch 230, loss 0.6882777088418437, correct 50, time 0.16 seconds
Epoch 240, loss 0.8148632679134246, correct 50, time 0.11 seconds
Epoch 250, loss 1.00958061866963, correct 50, time 0.12 seconds
Epoch 260, loss 0.11366305908169537, correct 50, time 0.11 seconds
Epoch 270, loss 0.009151233998635997, correct 50, time 0.11 seconds
Epoch 280, loss 0.33597955849704786, correct 49, time 0.12 seconds
Epoch 290, loss 0.016957209697545744, correct 50, time 0.11 seconds
Epoch 300, loss 0.026200380791581643, correct 50, time 0.12 seconds
Epoch 310, loss 0.6929734779508294, correct 50, time 0.11 seconds
Epoch 320, loss 0.24528007055171266, correct 50, time 0.14 seconds
Epoch 330, loss 0.2519082645030878, correct 49, time 0.14 seconds
Epoch 340, loss 0.6816702418345967, correct 50, time 0.14 seconds
Epoch 350, loss 0.34192934870541547, correct 50, time 0.14 seconds
Epoch 360, loss 0.2772153893972079, correct 50, time 0.14 seconds
Epoch 370, loss 0.8371572713872154, correct 50, time 0.14 seconds
Epoch 380, loss 0.02783097996665715, correct 50, time 0.25 seconds
Epoch 390, loss 0.12781339685904167, correct 50, time 0.15 seconds
Epoch 400, loss 0.3282797435560829, correct 50, time 0.15 seconds
Epoch 410, loss 0.006024770372831333, correct 50, time 0.14 seconds
Epoch 420, loss 0.2923961873816653, correct 50, time 0.15 seconds
Epoch 430, loss 0.019581193532563226, correct 50, time 0.14 seconds
Epoch 440, loss 0.017260245097213838, correct 50, time 0.15 seconds
Epoch 450, loss 0.29299085807796454, correct 50, time 0.15 seconds
Epoch 460, loss 0.24547205239473605, correct 50, time 0.14 seconds
Epoch 470, loss 0.09534738060121063, correct 50, time 0.14 seconds
Epoch 480, loss 0.3077533500410611, correct 50, time 0.15 seconds
Epoch 490, loss 0.098388512773383, correct 50, time 0.15 seconds


backend: gpu, dataset: simple, hidden: 100, rate: 0.05, points: 50 <br>
Epoch 0, loss 6.68296364997742, correct 37, time 17.74 seconds
Epoch 10, loss 2.383918154080912, correct 48, time 8.92 seconds
Epoch 20, loss 1.6429061701666883, correct 48, time 3.50 seconds
Epoch 30, loss 1.3679749965353651, correct 48, time 3.61 seconds
Epoch 40, loss 1.0135184895980613, correct 48, time 4.02 seconds
Epoch 50, loss 2.3424653219122815, correct 47, time 4.08 seconds
Epoch 60, loss 0.6325761445447473, correct 49, time 3.56 seconds
Epoch 70, loss 1.0453719903804104, correct 50, time 3.77 seconds
Epoch 80, loss 2.121157689763397, correct 46, time 3.46 seconds
Epoch 90, loss 0.5230149603322843, correct 50, time 3.50 seconds
Epoch 100, loss 0.4837384136625925, correct 50, time 3.89 seconds
Epoch 110, loss 0.8792224520712084, correct 50, time 3.88 seconds
Epoch 120, loss 1.0588779945104387, correct 48, time 4.17 seconds
Epoch 130, loss 0.3586506777623135, correct 50, time 3.62 seconds
Epoch 140, loss 0.0033141704232045365, correct 50, time 3.57 seconds
Epoch 150, loss 0.2159235573102668, correct 48, time 4.03 seconds
Epoch 160, loss 1.6465032568216644, correct 48, time 3.48 seconds
Epoch 170, loss 0.7556596802348093, correct 50, time 3.59 seconds
Epoch 180, loss 0.5330359323571452, correct 50, time 3.41 seconds
Epoch 190, loss 0.7515044807146463, correct 49, time 3.41 seconds
Epoch 200, loss 0.8452292745810825, correct 50, time 3.58 seconds
Epoch 210, loss 1.455229025610616, correct 49, time 3.54 seconds
Epoch 220, loss 0.587690827734966, correct 49, time 3.82 seconds
Epoch 230, loss 0.4220511836521446, correct 49, time 3.55 seconds
Epoch 240, loss 1.1336858393192348, correct 49, time 3.49 seconds
Epoch 250, loss 0.9979706738898009, correct 50, time 3.39 seconds
Epoch 260, loss 0.5632417980239892, correct 50, time 3.82 seconds
Epoch 270, loss 0.33583567129318836, correct 50, time 2.56 seconds
Epoch 280, loss 0.07653650579498168, correct 50, time 3.17 seconds
Epoch 290, loss 0.3331156186441163, correct 50, time 3.12 seconds
Epoch 300, loss 0.0024275010634724477, correct 50, time 3.33 seconds
Epoch 310, loss 0.6181014827323923, correct 50, time 2.66 seconds
Epoch 320, loss 0.570639450364238, correct 50, time 2.69 seconds
Epoch 330, loss 0.04300331304675153, correct 50, time 3.09 seconds
Epoch 340, loss 0.8986213637055306, correct 50, time 3.38 seconds
Epoch 350, loss 0.0011414751015550117, correct 49, time 3.63 seconds
Epoch 360, loss 0.42219524285311294, correct 50, time 3.35 seconds
Epoch 370, loss 0.18067900446251242, correct 50, time 3.22 seconds
Epoch 380, loss 0.15880222072823302, correct 50, time 3.65 seconds
Epoch 390, loss 1.5108025144555475, correct 48, time 3.43 seconds
Epoch 400, loss 0.46993625146845774, correct 50, time 3.42 seconds
Epoch 410, loss 0.19824938322973978, correct 50, time 2.62 seconds
Epoch 420, loss 0.19218040906332098, correct 49, time 2.87 seconds
Epoch 430, loss 0.4400570526457783, correct 50, time 2.64 seconds
Epoch 440, loss 0.3651940205557449, correct 50, time 3.03 seconds
Epoch 450, loss 1.0481747283537823, correct 49, time 2.88 seconds
Epoch 460, loss 0.2367450129668335, correct 50, time 2.73 seconds
Epoch 470, loss 0.024376564032230298, correct 50, time 2.79 seconds
Epoch 480, loss 0.2679060064414669, correct 50, time 2.57 seconds
Epoch 490, loss 0.3587546737625343, correct 50, time 2.55 seconds


backend: cpu, dataset: split, hidden: 100, rate: 0.05, points: 50<br>
Epoch 0, loss 9.19483399736696, correct 31, time 27.72 seconds
Epoch 10, loss 8.405148817681415, correct 39, time 0.11 seconds
Epoch 20, loss 5.383761765652042, correct 39, time 0.11 seconds
Epoch 30, loss 5.02391445705955, correct 43, time 0.13 seconds
Epoch 40, loss 5.541219299962146, correct 43, time 0.13 seconds
Epoch 50, loss 4.490757296141566, correct 37, time 0.11 seconds
Epoch 60, loss 4.972234363410205, correct 44, time 0.11 seconds
Epoch 70, loss 3.018466397654893, correct 42, time 0.11 seconds
Epoch 80, loss 4.546883788828131, correct 46, time 0.10 seconds
Epoch 90, loss 2.081491967384871, correct 46, time 0.10 seconds
Epoch 100, loss 1.6225524286219792, correct 48, time 0.11 seconds
Epoch 110, loss 2.379710088402255, correct 43, time 0.11 seconds
Epoch 120, loss 1.6126376006614547, correct 47, time 0.10 seconds
Epoch 130, loss 2.4098899788896366, correct 44, time 0.15 seconds
Epoch 140, loss 4.2603655270006415, correct 47, time 0.10 seconds
Epoch 150, loss 1.5852417438696964, correct 47, time 0.11 seconds
Epoch 160, loss 1.9129506162825392, correct 46, time 0.11 seconds
Epoch 170, loss 1.7246702595671435, correct 49, time 0.10 seconds
Epoch 180, loss 2.2362033187498143, correct 47, time 0.11 seconds
Epoch 190, loss 1.6565151370875626, correct 49, time 0.11 seconds
Epoch 200, loss 1.0867448301385187, correct 48, time 0.11 seconds
Epoch 210, loss 2.8660134963232737, correct 46, time 0.12 seconds
Epoch 220, loss 1.7080830178536939, correct 47, time 0.12 seconds
Epoch 230, loss 0.9837844659186151, correct 49, time 0.12 seconds
Epoch 240, loss 1.293236374146267, correct 45, time 0.11 seconds
Epoch 250, loss 1.4869302825292274, correct 46, time 0.12 seconds
Epoch 260, loss 0.8006648695748868, correct 48, time 0.12 seconds
Epoch 270, loss 1.9867720699013034, correct 47, time 0.12 seconds
Epoch 280, loss 2.6384880751299864, correct 49, time 0.11 seconds
Epoch 290, loss 3.4842850867453983, correct 44, time 0.11 seconds
Epoch 300, loss 1.9032442911441771, correct 49, time 0.11 seconds
Epoch 310, loss 1.0205146032244738, correct 49, time 0.11 seconds
Epoch 320, loss 0.2587598025608416, correct 49, time 0.11 seconds
Epoch 330, loss 1.9328023424670397, correct 49, time 0.11 seconds
Epoch 340, loss 0.2778828195423739, correct 49, time 0.11 seconds
Epoch 350, loss 0.5940669571030852, correct 49, time 0.11 seconds
Epoch 360, loss 2.5326698236915206, correct 49, time 0.11 seconds
Epoch 370, loss 0.5411868699100493, correct 49, time 0.13 seconds
Epoch 380, loss 0.9466621037952306, correct 49, time 0.24 seconds
Epoch 390, loss 1.0649361954061662, correct 48, time 0.13 seconds
Epoch 400, loss 0.7954955369284609, correct 49, time 0.14 seconds
Epoch 410, loss 0.7188432472201483, correct 47, time 0.14 seconds
Epoch 420, loss 0.4878156869012766, correct 46, time 0.14 seconds
Epoch 430, loss 3.0469188793542097, correct 47, time 0.14 seconds
Epoch 440, loss 1.3868048337030472, correct 48, time 0.16 seconds
Epoch 450, loss 2.345493699178271, correct 45, time 0.16 seconds
Epoch 460, loss 1.3453385303776253, correct 49, time 0.15 seconds
Epoch 470, loss 0.20964734374049118, correct 50, time 0.14 seconds
Epoch 480, loss 0.1614603977532976, correct 50, time 0.14 seconds
Epoch 490, loss 0.0118887780572651, correct 50, time 0.15 seconds

backend: gpu, dataset: split, hidden: 100, rate: 0.05, points: 50 <br>
Epoch 0, loss 5.657220562848061, correct 32, time 7.19 seconds
Epoch 10, loss 7.771813451817472, correct 20, time 3.59 seconds
Epoch 20, loss 4.390204274814756, correct 40, time 3.49 seconds
Epoch 30, loss 4.682450189826815, correct 45, time 3.60 seconds
Epoch 40, loss 2.4488682144433236, correct 48, time 3.49 seconds
Epoch 50, loss 2.4681724160292617, correct 49, time 3.56 seconds
Epoch 60, loss 2.4261143832587555, correct 45, time 3.70 seconds
Epoch 70, loss 1.7713401043514434, correct 50, time 3.59 seconds
Epoch 80, loss 1.603895997609591, correct 49, time 3.64 seconds
Epoch 90, loss 1.5644928653124834, correct 47, time 3.50 seconds
Epoch 100, loss 2.75301124400299, correct 48, time 3.59 seconds
Epoch 110, loss 1.539822924297964, correct 50, time 3.68 seconds
Epoch 120, loss 1.2809096437610408, correct 49, time 3.61 seconds
Epoch 130, loss 1.022715524899043, correct 50, time 3.56 seconds
Epoch 140, loss 1.5698478627537908, correct 49, time 3.45 seconds
Epoch 150, loss 0.4358144683444658, correct 50, time 3.51 seconds
Epoch 160, loss 0.9595922649696866, correct 50, time 3.46 seconds
Epoch 170, loss 0.5860427930078508, correct 50, time 3.60 seconds
Epoch 180, loss 0.38918018419997474, correct 49, time 3.44 seconds
Epoch 190, loss 0.45877751476919487, correct 50, time 3.52 seconds
Epoch 200, loss 1.3648233947085153, correct 50, time 3.61 seconds
Epoch 210, loss 0.029750519370476115, correct 49, time 3.54 seconds
Epoch 220, loss 0.38641139104088107, correct 49, time 3.64 seconds
Epoch 230, loss 0.613503799054451, correct 50, time 3.69 seconds
Epoch 240, loss 0.6863075324821715, correct 50, time 3.62 seconds
Epoch 250, loss 1.6095740382632653, correct 50, time 3.58 seconds
Epoch 260, loss 0.21085961251565635, correct 50, time 3.40 seconds
Epoch 270, loss 0.9586587702088661, correct 50, time 3.49 seconds
Epoch 280, loss 0.9764397941489014, correct 49, time 3.56 seconds
Epoch 290, loss 0.5618882048210706, correct 50, time 4.19 seconds
Epoch 300, loss 0.26327159872543693, correct 50, time 3.58 seconds
Epoch 310, loss 0.7917247891382888, correct 50, time 3.49 seconds
Epoch 320, loss 1.1096626014218196, correct 50, time 3.23 seconds
Epoch 330, loss 0.6406320778971238, correct 50, time 2.69 seconds
Epoch 340, loss 0.846313813545545, correct 50, time 3.32 seconds
Epoch 350, loss 0.7583041295213265, correct 50, time 3.26 seconds
Epoch 360, loss 0.22940401146484063, correct 50, time 3.26 seconds
Epoch 370, loss 0.49462464196994654, correct 50, time 3.08 seconds
Epoch 380, loss 0.03286338576384899, correct 50, time 2.48 seconds
Epoch 390, loss 0.25490411655417533, correct 50, time 3.22 seconds
Epoch 400, loss 0.010084630896794147, correct 50, time 3.20 seconds
Epoch 410, loss 0.6792887063297882, correct 50, time 2.48 seconds
Epoch 420, loss 0.3898809696778756, correct 50, time 2.55 seconds
Epoch 430, loss 0.06365866209141839, correct 50, time 2.81 seconds
Epoch 440, loss 0.19875353335078066, correct 50, time 2.53 seconds
Epoch 450, loss 0.8519215056759522, correct 50, time 2.56 seconds
Epoch 460, loss 0.6238691580975493, correct 50, time 2.56 seconds
Epoch 470, loss 0.08788143184323322, correct 50, time 3.88 seconds
Epoch 480, loss 0.10617325375419831, correct 50, time 3.98 seconds
Epoch 490, loss 0.6531245361606888, correct 50, time 3.43 seconds

backend: cpu, dataset: xor, hidden: 100, rate: 0.05, points: 50 <br>
Epoch 0, loss 6.419373459460569, correct 27, time 32.09 seconds
Epoch 10, loss 6.96875208904579, correct 42, time 0.11 seconds
Epoch 20, loss 2.8649128518784948, correct 45, time 0.11 seconds
Epoch 30, loss 3.550069376631923, correct 44, time 0.15 seconds
Epoch 40, loss 3.5270420886898934, correct 46, time 0.11 seconds
Epoch 50, loss 1.4851711271250518, correct 46, time 0.15 seconds
Epoch 60, loss 5.155502500650663, correct 44, time 0.18 seconds
Epoch 70, loss 3.5449007590415262, correct 46, time 0.13 seconds
Epoch 80, loss 2.665217641898387, correct 46, time 0.16 seconds
Epoch 90, loss 1.8992136017299808, correct 47, time 0.14 seconds
Epoch 100, loss 1.304903268343776, correct 46, time 0.12 seconds
Epoch 110, loss 1.7048108392893226, correct 47, time 0.13 seconds
Epoch 120, loss 1.7524713135351644, correct 47, time 0.13 seconds
Epoch 130, loss 0.7800397254325747, correct 47, time 0.14 seconds
Epoch 140, loss 1.3942308970284822, correct 48, time 0.16 seconds
Epoch 150, loss 1.2982502268270872, correct 47, time 0.16 seconds
Epoch 160, loss 2.1938609893261294, correct 47, time 0.15 seconds
Epoch 170, loss 2.0435174243687046, correct 49, time 0.15 seconds
Epoch 180, loss 2.0413721360855894, correct 47, time 0.15 seconds
Epoch 190, loss 0.5956757905091553, correct 46, time 0.11 seconds
Epoch 200, loss 0.36561658763707117, correct 50, time 0.12 seconds
Epoch 210, loss 0.8991228722968455, correct 50, time 0.12 seconds
Epoch 220, loss 1.7723949286574074, correct 50, time 0.13 seconds
Epoch 230, loss 0.29513737368708715, correct 48, time 0.11 seconds
Epoch 240, loss 1.4623761056995532, correct 50, time 0.15 seconds
Epoch 250, loss 0.8397449821010389, correct 48, time 0.15 seconds
Epoch 260, loss 2.6216743801285287, correct 46, time 0.12 seconds
Epoch 270, loss 1.184200650942846, correct 50, time 0.11 seconds
Epoch 280, loss 1.6473918764442097, correct 49, time 0.11 seconds
Epoch 290, loss 0.8798826080569603, correct 50, time 0.12 seconds
Epoch 300, loss 2.5755984258730074, correct 48, time 0.11 seconds
Epoch 310, loss 1.2145702502093394, correct 50, time 0.11 seconds
Epoch 320, loss 0.6204167470151246, correct 50, time 0.14 seconds
Epoch 330, loss 0.7355583584759526, correct 49, time 0.27 seconds
Epoch 340, loss 0.868971228440603, correct 50, time 0.19 seconds
Epoch 350, loss 0.8815149351708758, correct 50, time 0.18 seconds
Epoch 360, loss 1.2464426577660854, correct 50, time 0.19 seconds
Epoch 370, loss 0.6491648972716044, correct 50, time 0.19 seconds
Epoch 380, loss 0.11424263970568796, correct 50, time 0.31 seconds
Epoch 390, loss 0.14650772760751887, correct 49, time 0.19 seconds
Epoch 400, loss 0.8899271349923312, correct 50, time 0.21 seconds
Epoch 410, loss 1.4237794193256832, correct 49, time 0.19 seconds
Epoch 420, loss 0.03991044124805683, correct 50, time 0.20 seconds
Epoch 430, loss 0.55792539173454, correct 50, time 0.15 seconds
Epoch 440, loss 0.3313704762488577, correct 50, time 0.17 seconds
Epoch 450, loss 0.6332956089620122, correct 50, time 0.14 seconds
Epoch 460, loss 0.8425150715231002, correct 50, time 0.20 seconds
Epoch 470, loss 1.2688608887803634, correct 50, time 0.14 seconds
Epoch 480, loss 0.8351903102009907, correct 50, time 0.14 seconds
Epoch 490, loss 0.04955592718754653, correct 50, time 0.14 seconds

backend: gpu, dataset: xor, hidden: 100, rate: 0.05, points: 50 <nr>
Epoch 0, loss 7.476310475180543, correct 30, time 7.16 seconds
Epoch 10, loss 4.727476276456226, correct 43, time 3.23 seconds
Epoch 20, loss 4.413027300862569, correct 41, time 3.22 seconds
Epoch 30, loss 3.1480314596256616, correct 44, time 3.21 seconds
Epoch 40, loss 2.6570283062155857, correct 48, time 3.45 seconds
Epoch 50, loss 4.438296014371257, correct 48, time 3.44 seconds
Epoch 60, loss 1.6032287139331776, correct 48, time 3.39 seconds
Epoch 70, loss 1.4258468236981876, correct 47, time 3.16 seconds
Epoch 80, loss 1.288726491884274, correct 47, time 4.18 seconds
Epoch 90, loss 3.2911100403264753, correct 47, time 3.98 seconds
Epoch 100, loss 0.7621658342723722, correct 47, time 3.96 seconds
Epoch 110, loss 1.571021897498789, correct 48, time 4.22 seconds
Epoch 120, loss 0.6844930235280375, correct 49, time 3.69 seconds
Epoch 130, loss 1.8801416915272575, correct 50, time 4.02 seconds
Epoch 140, loss 1.1411761922931292, correct 50, time 3.38 seconds
Epoch 150, loss 2.4027336608925967, correct 47, time 4.15 seconds
Epoch 160, loss 1.7413176130089363, correct 47, time 2.98 seconds
Epoch 170, loss 1.0501054598131099, correct 50, time 3.60 seconds
Epoch 180, loss 1.0586521114711107, correct 50, time 4.06 seconds
Epoch 190, loss 0.8751752813658673, correct 49, time 3.57 seconds
Epoch 200, loss 1.0466373827578237, correct 49, time 3.62 seconds
Epoch 210, loss 0.5148694853179019, correct 48, time 3.87 seconds
Epoch 220, loss 0.5572465701582537, correct 50, time 3.53 seconds
Epoch 230, loss 1.7344150305517751, correct 50, time 3.58 seconds
Epoch 240, loss 0.522941122367351, correct 50, time 3.77 seconds
Epoch 250, loss 0.9856003388654584, correct 50, time 3.91 seconds
Epoch 260, loss 0.5834094350937776, correct 49, time 3.79 seconds
Epoch 270, loss 0.5554515785431535, correct 50, time 3.52 seconds
Epoch 280, loss 1.1232493392709912, correct 50, time 3.45 seconds
Epoch 290, loss 0.22963831982288518, correct 50, time 3.71 seconds
Epoch 300, loss 0.6580740063003754, correct 49, time 3.91 seconds
Epoch 310, loss 1.1141547967286414, correct 50, time 4.03 seconds
Epoch 320, loss 0.29045352192316587, correct 50, time 3.95 seconds
Epoch 330, loss 0.466213687759308, correct 50, time 3.94 seconds
Epoch 340, loss 0.29610189286700955, correct 50, time 3.71 seconds
Epoch 350, loss 0.5271501601121608, correct 50, time 3.60 seconds
Epoch 360, loss 0.4457292973818425, correct 50, time 4.08 seconds
Epoch 370, loss 1.0870483786650893, correct 50, time 3.77 seconds
Epoch 380, loss 0.1411549803445361, correct 50, time 3.57 seconds
Epoch 390, loss 0.4501398033716228, correct 50, time 3.65 seconds
Epoch 400, loss 0.43881592090913646, correct 50, time 3.55 seconds
Epoch 410, loss 0.2523778506958985, correct 50, time 3.31 seconds
Epoch 420, loss 0.6947465871763814, correct 50, time 3.44 seconds
Epoch 430, loss 0.13551512946304572, correct 50, time 4.09 seconds
Epoch 440, loss 0.15171721673933064, correct 50, time 3.99 seconds
Epoch 450, loss 0.20444077374456135, correct 50, time 3.62 seconds
Epoch 460, loss 0.05864154473204009, correct 50, time 3.18 seconds
Epoch 470, loss 0.14762431192793155, correct 50, time 4.14 seconds
Epoch 480, loss 0.5213497853652733, correct 50, time 4.17 seconds
Epoch 490, loss 0.2111923492298178, correct 50, time 3.89 seconds

# Large Model
backend: cpu, dataset: simple, hidden: 250, rate: 0.05, points: 50<br>
Epoch 0, loss 0.7606393531559354, correct 49, time 35.80 seconds
Epoch 10, loss 0.25941216342984647, correct 50, time 0.53 seconds
Epoch 20, loss 0.479705376807187, correct 48, time 0.41 seconds
Epoch 30, loss 0.24701770173009144, correct 50, time 0.40 seconds
Epoch 40, loss 0.29156875549110806, correct 48, time 0.41 seconds
Epoch 50, loss 0.015396037252339529, correct 50, time 0.42 seconds
Epoch 60, loss 0.13417694547685458, correct 50, time 0.42 seconds
Epoch 70, loss 0.22460687213152644, correct 50, time 0.49 seconds
Epoch 80, loss 0.0023624080640160397, correct 50, time 0.59 seconds
Epoch 90, loss 0.5129697161731503, correct 50, time 0.53 seconds
Epoch 100, loss 0.3444883567458985, correct 50, time 0.54 seconds
Epoch 110, loss 0.10833108221723622, correct 50, time 0.64 seconds
Epoch 120, loss 0.040486322647658855, correct 50, time 0.55 seconds
Epoch 130, loss 0.35612188517581767, correct 50, time 0.53 seconds
Epoch 140, loss 0.3062603012960327, correct 50, time 0.58 seconds
Epoch 150, loss 0.19370745330178601, correct 50, time 0.50 seconds
Epoch 160, loss 0.13267991882144273, correct 50, time 0.44 seconds
Epoch 170, loss 0.17216288936913424, correct 50, time 0.42 seconds
Epoch 180, loss 0.01887248989983529, correct 50, time 0.46 seconds
Epoch 190, loss 0.27022015862506515, correct 50, time 0.46 seconds
Epoch 200, loss 0.022726344496494036, correct 50, time 0.40 seconds
Epoch 210, loss 0.22376940465980433, correct 50, time 0.42 seconds
Epoch 220, loss 0.002833265926793888, correct 50, time 0.46 seconds
Epoch 230, loss 0.13874102905349997, correct 50, time 0.54 seconds
Epoch 240, loss 0.00033391868919162797, correct 50, time 0.52 seconds
Epoch 250, loss 0.24121215958688833, correct 50, time 0.52 seconds
Epoch 260, loss 0.008829711518609925, correct 50, time 0.57 seconds
Epoch 270, loss 0.05880268486726747, correct 50, time 0.59 seconds
Epoch 280, loss 0.12900200224119707, correct 50, time 0.58 seconds
Epoch 290, loss 0.13362182892094088, correct 50, time 0.53 seconds
Epoch 300, loss 0.10495298734942801, correct 50, time 0.48 seconds
Epoch 310, loss 0.30206491752603387, correct 50, time 0.51 seconds
Epoch 320, loss 0.0005357039948453944, correct 50, time 0.45 seconds
Epoch 330, loss 0.07813700592831253, correct 50, time 0.44 seconds
Epoch 340, loss 0.13807468597441142, correct 50, time 0.43 seconds
Epoch 350, loss 0.2549917607494568, correct 50, time 0.43 seconds
Epoch 360, loss 0.004676846760506378, correct 50, time 0.43 seconds
Epoch 370, loss 0.005057016980271147, correct 50, time 0.46 seconds
Epoch 380, loss 0.06302449143667904, correct 50, time 0.60 seconds
Epoch 390, loss 0.10731689785920978, correct 50, time 0.49 seconds
Epoch 400, loss 0.04661558585274219, correct 50, time 0.54 seconds
Epoch 410, loss 0.21511055621319977, correct 50, time 0.56 seconds
Epoch 420, loss 0.0684631396731972, correct 50, time 0.50 seconds
Epoch 430, loss 0.003447190149620545, correct 50, time 0.45 seconds
Epoch 440, loss 0.03425895657445207, correct 50, time 0.55 seconds
Epoch 450, loss 0.09722884211891007, correct 50, time 0.44 seconds
Epoch 460, loss 0.000541937981507568, correct 50, time 0.45 seconds
Epoch 470, loss 9.586667429816694e-05, correct 50, time 0.44 seconds
Epoch 480, loss -4.645286439911455e-06, correct 50, time 0.41 seconds
Epoch 490, loss 0.14047485034656124, correct 50, time 0.40 seconds

backend: gpu, dataset: simple, hidden: 250, rate: 0.05, points: 50 <br>
Epoch 0, loss 8.949765890057147, correct 44, time 7.30 seconds
Epoch 10, loss 1.1780632152832902, correct 46, time 3.68 seconds
Epoch 20, loss 1.193418335251185, correct 48, time 7.53 seconds
Epoch 30, loss 0.6611865574924014, correct 50, time 7.64 seconds
Epoch 40, loss 1.1058636676993872, correct 48, time 7.68 seconds
Epoch 50, loss 1.4334337108619828, correct 48, time 7.68 seconds
Epoch 60, loss 0.38200889250764447, correct 50, time 8.24 seconds
Epoch 70, loss 0.41254783743161483, correct 50, time 8.72 seconds
Epoch 80, loss 0.45307135126679776, correct 50, time 7.74 seconds
Epoch 90, loss 0.2938271411180099, correct 50, time 8.53 seconds
Epoch 100, loss 0.4660227088237348, correct 50, time 11.32 seconds
Epoch 110, loss 0.174378309327718, correct 50, time 8.70 seconds
Epoch 120, loss 0.04372808799052591, correct 50, time 7.45 seconds
Epoch 130, loss 0.24977199911236714, correct 50, time 9.87 seconds
Epoch 140, loss 0.14940814870846883, correct 50, time 11.10 seconds
Epoch 150, loss 0.15723165075658216, correct 50, time 8.60 seconds
Epoch 160, loss 0.2396529567130677, correct 50, time 7.83 seconds
Epoch 170, loss 0.39917608407227373, correct 50, time 7.59 seconds
Epoch 180, loss 0.1736258876511395, correct 50, time 8.55 seconds
Epoch 190, loss 0.009911098870725533, correct 50, time 8.54 seconds
Epoch 200, loss 0.14869462847652706, correct 50, time 7.91 seconds
Epoch 210, loss 0.052436858503119375, correct 50, time 10.12 seconds
Epoch 220, loss 0.038385657114081415, correct 50, time 8.13 seconds
Epoch 230, loss 0.18380601123068963, correct 50, time 7.90 seconds
Epoch 240, loss 0.006375664141886459, correct 50, time 9.28 seconds
Epoch 250, loss 0.06424960574778933, correct 50, time 8.17 seconds
Epoch 260, loss 0.12280402621920568, correct 50, time 7.67 seconds
Epoch 270, loss 0.0005924639157333649, correct 50, time 8.31 seconds
Epoch 280, loss 0.07583529258635491, correct 50, time 8.00 seconds
Epoch 290, loss 0.00899937371260089, correct 50, time 8.80 seconds
Epoch 300, loss 0.15960871526697817, correct 50, time 9.55 seconds
Epoch 310, loss 0.08227870855260512, correct 50, time 7.93 seconds
Epoch 320, loss 0.051363096728270656, correct 50, time 9.00 seconds
Epoch 330, loss 0.09642737022999776, correct 50, time 7.84 seconds
Epoch 340, loss 0.059512855315128996, correct 50, time 8.07 seconds
Epoch 350, loss 0.13638312400410646, correct 50, time 7.31 seconds
Epoch 360, loss 0.011799600290026626, correct 50, time 7.39 seconds
Epoch 370, loss 0.053246553912188084, correct 50, time 7.84 seconds
Epoch 380, loss 0.059918990807807854, correct 50, time 7.78 seconds
Epoch 390, loss 0.051330225943388826, correct 50, time 7.62 seconds
Epoch 400, loss 0.047674179686858276, correct 50, time 7.36 seconds
Epoch 410, loss 0.053369120534026634, correct 50, time 7.93 seconds
Epoch 420, loss 0.029081420695874428, correct 50, time 7.88 seconds
Epoch 430, loss 0.1402748098568458, correct 50, time 7.80 seconds
Epoch 440, loss 0.04720081160605938, correct 50, time 8.26 seconds
Epoch 450, loss 0.01216883762383722, correct 50, time 8.45 seconds
Epoch 460, loss 0.029793759562434213, correct 50, time 7.66 seconds
Epoch 470, loss 0.03399455811805155, correct 50, time 7.80 seconds
Epoch 480, loss 0.01759348859451171, correct 50, time 8.45 seconds
Epoch 490, loss 0.09285372876842674, correct 50, time 7.58 seconds