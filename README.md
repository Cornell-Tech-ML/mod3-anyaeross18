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


# 3.1 & 3.2 Parallel Analytics Script <br>
## MAP

        ================================================================================
        Parallel Accelerator Optimizing: Function tensor_map.<locals>._map,
        C:\Users\15184\OneDrive\Documents\Machine Learning
        Engineering\mod3-anyaeross18\minitorch\fast_ops.py (174)
        ================================================================================

        Parallel loop listing for Function tensor_map.<locals>._map,
        C:\Users\15184\OneDrive\Documents\Machine Learning Engineering\mod3-anyaeross18\minitorch\fast_ops.py (174)
        -----------------------------------------------------------------------------|loop #ID
        def _map(                                                                |
                out: Storage,                                                        |
                out_shape: Shape,                                                    |
                out_strides: Strides,                                                |
                in_storage: Storage,                                                 |
                in_shape: Shape,                                                     |
                in_strides: Strides,                                                 |
        ) -> None:                                                               |
                if len(out_strides) != len(in_strides):                              |
                strides_match = False                                            |
                elif (out_strides != in_strides).any():------------------------------| #0
                strides_match = False                                            |
                elif (out_shape != in_shape).any():----------------------------------| #1
                strides_match = False                                            |
                else:                                                                |
                strides_match = True                                             |
                                                                                |
                if strides_match:                                                    |
                for i in prange(out.size):---------------------------------------| #2
                        out[i] = fn(in_storage[i])                                   |
                else:                                                                |
                for i in prange(out.size):---------------------------------------| #3
                        out_index = np.zeros_like(out_shape, dtype=np.int32)         |
                        in_index = np.zeros_like(in_shape, dtype=np.int32)           |
                        to_index(i, out_shape, out_index)                            |
                        broadcast_index(out_index, out_shape, in_shape, in_index)    |
                        position_in = index_to_position(in_index, in_strides)        |
                        position_out = index_to_position(out_index, out_strides)     |
                        out[position_out] = fn(in_storage[position_in])              |
        --------------------------------- Fusing loops ---------------------------------
        Attempting fusion of parallel loops (combines loops with similar properties)...
        Following the attempted fusion of parallel for-loops there are 4 parallel for-
        loop(s) (originating from loops labelled: #0, #1, #2, #3).
        --------------------------------------------------------------------------------
        ----------------------------- Before Optimisation ------------------------------
        --------------------------------------------------------------------------------
        ------------------------------ After Optimisation ------------------------------
        Parallel structure is already optimal.
        --------------------------------------------------------------------------------
        --------------------------------------------------------------------------------

        ---------------------------Loop invariant code motion---------------------------
        Allocation hoisting:
        No allocation hoisting found
        None


## ZIP

        ================================================================================
        Parallel Accelerator Optimizing: Function tensor_zip.<locals>._zip,
        C:\Users\15184\OneDrive\Documents\Machine Learning
        Engineering\mod3-anyaeross18\minitorch\fast_ops.py (230)
        ================================================================================

        Parallel loop listing for Function tensor_zip.<locals>._zip, C:\Users\15184\OneDrive\Documents\Machine Learning Engineering\mod3-anyaeross18\minitorch\fast_ops.py (230)
        ----------------------------------------------------------------------------------------|loop #ID
        def _zip(                                                                           |
                out: Storage,                                                                   |
                out_shape: Shape,                                                               |
                out_strides: Strides,                                                           |
                a_storage: Storage,                                                             |
                a_shape: Shape,                                                                 |
                a_strides: Strides,                                                             |
                b_storage: Storage,                                                             |
                b_shape: Shape,                                                                 |
                b_strides: Strides,                                                             |
        ) -> None:                                                                          |
                if len(out_strides) != len(a_strides):                                          |
                strides_match = False                                                       |
                elif len(out_strides) != len(b_strides):                                        |
                strides_match = False                                                       |
                elif (out_strides != a_strides).any():------------------------------------------| #4
                strides_match = False                                                       |
                elif (out_strides != b_strides).any():------------------------------------------| #5
                strides_match = False                                                       |
                elif (out_shape != a_shape).any():----------------------------------------------| #6
                strides_match = False                                                       |
                elif (out_shape != b_shape).any():----------------------------------------------| #7
                strides_match = False                                                       |
                else:                                                                           |
                strides_match = True                                                        |
                                                                                                |
                # If strides match, apply the function directly                                 |
                if strides_match:                                                               |
                for i in prange(out.size):--------------------------------------------------| #8
                        out[i] = fn(a_storage[i], b_storage[i])                                 |
                else:                                                                           |
                for i in prange(out.size):--------------------------------------------------| #9
                        out_index = np.zeros_like(out_shape, dtype=np.int32)                    |
                        a_index = np.zeros_like(a_shape, dtype=np.int32)                        |
                        b_index = np.zeros_like(b_shape, dtype=np.int32)                        |
                                                                                                |
                        to_index(i, out_shape, out_index)                                       |
                        position_out = index_to_position(out_index, out_strides)                |
                        broadcast_index(out_index, out_shape, a_shape, a_index)                 |
                        position_a = index_to_position(a_index, a_strides)                      |
                        broadcast_index(out_index, out_shape, b_shape, b_index)                 |
                        position_b = index_to_position(b_index, b_strides)                      |
                        out[position_out] = fn(a_storage[position_a], b_storage[position_b])    |
        --------------------------------- Fusing loops ---------------------------------
        Attempting fusion of parallel loops (combines loops with similar properties)...
        Following the attempted fusion of parallel for-loops there are 6 parallel for-
        loop(s) (originating from loops labelled: #4, #5, #6, #7, #8, #9).
        --------------------------------------------------------------------------------
        ----------------------------- Before Optimisation ------------------------------
        --------------------------------------------------------------------------------
        ------------------------------ After Optimisation ------------------------------
        Parallel structure is already optimal.
        --------------------------------------------------------------------------------
        --------------------------------------------------------------------------------

        ---------------------------Loop invariant code motion---------------------------
        Allocation hoisting:
        No allocation hoisting found
        None


## REDUCE

        ================================================================================
        Parallel Accelerator Optimizing: Function tensor_reduce.<locals>._reduce,
        C:\Users\15184\OneDrive\Documents\Machine Learning
        Engineering\mod3-anyaeross18\minitorch\fast_ops.py (298)
        ================================================================================

        Parallel loop listing for Function tensor_reduce.<locals>._reduce, C:\Users\15184\OneDrive\Documents\Machine Learning Engineering\mod3-anyaeross18\minitorch\fast_ops.py (298)
        ----------------------------------------------------------------------------|loop #ID
        def _reduce(                                                            |
                out: Storage,                                                       |
                out_shape: Shape,                                                   |
                out_strides: Strides,                                               |
                a_storage: Storage,                                                 |
                a_shape: Shape,                                                     |
                a_strides: Strides,                                                 |
                reduce_dim: int,                                                    |
        ) -> None:                                                              |
                # reduce_size = a_shape[reduce_dim]                                 |
                for i in prange(out.size):------------------------------------------| #10
                out_index = np.zeros_like(out_shape, dtype=np.int32)            |
                # reduce_size = a_shape[reduce_dim]                             |
                to_index(i, out_shape, out_index)                               |
                o = index_to_position(out_index, out_strides)                   |
                position_a_base = index_to_position(out_index, a_strides)       |
                temp = a_storage[position_a_base]                               |
                for j in range(1, a_shape[reduce_dim]):                         |
                        position_a = position_a_base + j * a_strides[reduce_dim]    |
                        temp = fn(temp, float(a_storage[position_a]))               |
                out[o] = temp                                                   |
        --------------------------------- Fusing loops ---------------------------------
        Attempting fusion of parallel loops (combines loops with similar properties)...
        Following the attempted fusion of parallel for-loops there are 1 parallel for-
        loop(s) (originating from loops labelled: #10).
        --------------------------------------------------------------------------------
        ----------------------------- Before Optimisation ------------------------------
        --------------------------------------------------------------------------------
        ------------------------------ After Optimisation ------------------------------
        Parallel structure is already optimal.
        --------------------------------------------------------------------------------
        --------------------------------------------------------------------------------

        ---------------------------Loop invariant code motion---------------------------
        Allocation hoisting:
        No allocation hoisting found
        None

## MATRIX MULTIPLY

        ================================================================================
        Parallel Accelerator Optimizing: Function _tensor_matrix_multiply,
        C:\Users\15184\OneDrive\Documents\Machine Learning
        Engineering\mod3-anyaeross18\minitorch\fast_ops.py (323)
        ================================================================================

        Parallel loop listing for Function _tensor_matrix_multiply, C:\Users\15184\OneDrive\Documents\Machine Learning Engineering\mod3-anyaeross18\minitorch\fast_ops.py (323)
        -------------------------------------------------------------------------------------|loop #ID
        def _tensor_matrix_multiply(                                                         |
        out: Storage,                                                                    |
        out_shape: Shape,                                                                |
        out_strides: Strides,                                                            |
        a_storage: Storage,                                                              |
        a_shape: Shape,                                                                  |
        a_strides: Strides,                                                              |
        b_storage: Storage,                                                              |
        b_shape: Shape,                                                                  |
        b_strides: Strides,                                                              |
        ) -> None:                                                                           |
        """NUMBA tensor matrix multiply function.                                        |
                                                                                        |
        Should work for any tensor shapes that broadcast as long as                      |
                                                                                        |
        ```                                                                              |
        assert a_shape[-1] == b_shape[-2]                                                |
        ```                                                                              |
                                                                                        |
        Optimizations:                                                                   |
                                                                                        |
        * Outer loop in parallel                                                         |
        * No index buffers or function calls                                             |
        * Inner loop should have no global writes, 1 multiply.                           |
                                                                                        |
                                                                                        |
        Args:                                                                            |
        ----                                                                             |
                out (Storage): storage for `out` tensor                                      |
                out_shape (Shape): shape for `out` tensor                                    |
                out_strides (Strides): strides for `out` tensor                              |
                a_storage (Storage): storage for `a` tensor                                  |
                a_shape (Shape): shape for `a` tensor                                        |
                a_strides (Strides): strides for `a` tensor                                  |
                b_storage (Storage): storage for `b` tensor                                  |
                b_shape (Shape): shape for `b` tensor                                        |
                b_strides (Strides): strides for `b` tensor                                  |
                                                                                        |
        Returns:                                                                         |
        -------                                                                          |
                None : Fills in `out`                                                        |
                                                                                        |
        """                                                                              |
        assert a_shape[-1] == b_shape[-2]                                                |
        a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                           |
        b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                           |
                                                                                        |
        for i in prange(out.size):-------------------------------------------------------| #11
                out0 = i // (out_shape[-1] * out_shape[-2])                                  |
                out1 = (i % (out_shape[-1] * out_shape[-2])) // out_shape[-1]                |
                out2 = i % out_shape[-1]                                                     |
                                                                                        |
                out_index = (                                                                |
                out0 * out_strides[0] + out1 * out_strides[1] + out2 * out_strides[2]    |
                )                                                                            |
                a_start = out0 * a_batch_stride + out1 * a_strides[-2]                       |
                b_start = out0 * b_batch_stride + out2 * b_strides[-1]                       |
                                                                                        |
                temp = 0.0                                                                   |
                for j in range(a_shape[-1]):                                                 |
                a_index = a_start + j * a_strides[-1]                                    |
                b_index = b_start + j * b_strides[-2]                                    |
                temp += a_storage[a_index] * b_storage[b_index]                          |
                out[out_index] = temp                                                        |
        --------------------------------- Fusing loops ---------------------------------
        Attempting fusion of parallel loops (combines loops with similar properties)...
        Following the attempted fusion of parallel for-loops there are 1 parallel for-
        loop(s) (originating from loops labelled: #11).
        --------------------------------------------------------------------------------
        ----------------------------- Before Optimisation ------------------------------
        --------------------------------------------------------------------------------
        ------------------------------ After Optimisation ------------------------------
        Parallel structure is already optimal.
        --------------------------------------------------------------------------------
        --------------------------------------------------------------------------------

        ---------------------------Loop invariant code motion---------------------------
        Allocation hoisting:
        No allocation hoisting found
        None


# Timing Graph 3.4:<br>

## Timing summary:<br>
Size: 64
    fast: 0.00351
    gpu: 0.01174
Size: 128
    fast: 0.01696
    gpu: 0.02465
Size: 256
    fast: 0.07248
    gpu: 0.11462
Size: 512
    fast: 0.37120
    gpu: 0.36778
Size: 1024
    fast: 3.67718
    gpu: 1.41245

## Graph:<br>


# 3.5: Training Loss, Accuracy, and Timing<br>
## Small Models:<br>
backend: cpu, dataset: simple, hidden: 100, rate: 0.05, points: 50 <br>
Epoch 0, loss 5.52817496286788, correct 27, time 30.40 seconds<br>
Epoch 10, loss 2.705329348675475, correct 47, time 0.12 seconds<br>
Epoch 20, loss 1.4890510000659218, correct 48, time 0.11 seconds<br>
Epoch 30, loss 0.8635561816490419, correct 49, time 0.13 seconds<br>
Epoch 40, loss 2.102966128787591, correct 50, time 0.11 seconds<br>
Epoch 50, loss 0.3520923121836196, correct 50, time 0.11 seconds<br>
Epoch 60, loss 0.9942342939645927, correct 49, time 0.12 seconds<br>
Epoch 70, loss 0.5556668354600687, correct 49, time 0.12 seconds<br>
Epoch 80, loss 0.8704323319032629, correct 49, time 0.15 seconds<br>
Epoch 90, loss 0.41930459163684614, correct 49, time 0.15 seconds<br>
Epoch 100, loss 0.38002522357595137, correct 50, time 0.12 seconds<br>
Epoch 110, loss 0.22176702976972645, correct 50, time 0.11 seconds<br>
Epoch 120, loss 0.15380724550617056, correct 50, time 0.12 seconds<br>
Epoch 130, loss 0.9830008389780323, correct 49, time 0.14 seconds<br>
Epoch 140, loss 0.8130525341225274, correct 50, time 0.12 seconds<br>
Epoch 150, loss 0.9425638691990154, correct 50, time 0.11 seconds<br>
Epoch 160, loss 0.03902290282321021, correct 50, time 0.14 seconds<br>
Epoch 170, loss 0.5649559134783091, correct 50, time 0.11 seconds<br>
Epoch 180, loss 0.11346897367239869, correct 50, time 0.11 seconds<br>
Epoch 190, loss 0.009645903830819764, correct 50, time 0.13 seconds<br>
Epoch 200, loss 0.7143023831890992, correct 50, time 0.15 seconds<br>
Epoch 210, loss 0.030674413706201026, correct 50, time 0.11 seconds<br>
Epoch 220, loss 0.01696839361679683, correct 50, time 0.16 seconds<br>
Epoch 230, loss 0.6882777088418437, correct 50, time 0.16 seconds<br>
Epoch 240, loss 0.8148632679134246, correct 50, time 0.11 seconds<br>
Epoch 250, loss 1.00958061866963, correct 50, time 0.12 seconds<br>
Epoch 260, loss 0.11366305908169537, correct 50, time 0.11 seconds<br>
Epoch 270, loss 0.009151233998635997, correct 50, time 0.11 seconds<br>
Epoch 280, loss 0.33597955849704786, correct 49, time 0.12 seconds<br>
Epoch 290, loss 0.016957209697545744, correct 50, time 0.11 seconds<br>
Epoch 300, loss 0.026200380791581643, correct 50, time 0.12 seconds<br>
Epoch 310, loss 0.6929734779508294, correct 50, time 0.11 seconds<br>
Epoch 320, loss 0.24528007055171266, correct 50, time 0.14 seconds<br>
Epoch 330, loss 0.2519082645030878, correct 49, time 0.14 seconds<br>
Epoch 340, loss 0.6816702418345967, correct 50, time 0.14 seconds<br>
Epoch 350, loss 0.34192934870541547, correct 50, time 0.14 seconds<br>
Epoch 360, loss 0.2772153893972079, correct 50, time 0.14 seconds<br>
Epoch 370, loss 0.8371572713872154, correct 50, time 0.14 seconds<br>
Epoch 380, loss 0.02783097996665715, correct 50, time 0.25 seconds<br>
Epoch 390, loss 0.12781339685904167, correct 50, time 0.15 seconds<br>
Epoch 400, loss 0.3282797435560829, correct 50, time 0.15 seconds<br>
Epoch 410, loss 0.006024770372831333, correct 50, time 0.14 seconds<br>
Epoch 420, loss 0.2923961873816653, correct 50, time 0.15 seconds<br>
Epoch 430, loss 0.019581193532563226, correct 50, time 0.14 seconds<br>
Epoch 440, loss 0.017260245097213838, correct 50, time 0.15 seconds<br>
Epoch 450, loss 0.29299085807796454, correct 50, time 0.15 seconds<br>
Epoch 460, loss 0.24547205239473605, correct 50, time 0.14 seconds<br>
Epoch 470, loss 0.09534738060121063, correct 50, time 0.14 seconds<br>
Epoch 480, loss 0.3077533500410611, correct 50, time 0.15 seconds<br>
Epoch 490, loss 0.098388512773383, correct 50, time 0.15 seconds<br>

backend: gpu, dataset: simple, hidden: 100, rate: 0.05, points: 50 <br>
Epoch 0, loss 6.68296364997742, correct 37, time 17.74 seconds<br>
Epoch 10, loss 2.383918154080912, correct 48, time 8.92 seconds<br>
Epoch 20, loss 1.6429061701666883, correct 48, time 3.50 seconds<br>
Epoch 30, loss 1.3679749965353651, correct 48, time 3.61 seconds<br>
Epoch 40, loss 1.0135184895980613, correct 48, time 4.02 seconds<br>
Epoch 50, loss 2.3424653219122815, correct 47, time 4.08 seconds<br>
Epoch 60, loss 0.6325761445447473, correct 49, time 3.56 seconds<br>
Epoch 70, loss 1.0453719903804104, correct 50, time 3.77 seconds<br>
Epoch 80, loss 2.121157689763397, correct 46, time 3.46 seconds<br>
Epoch 90, loss 0.5230149603322843, correct 50, time 3.50 seconds<br>
Epoch 100, loss 0.4837384136625925, correct 50, time 3.89 seconds<br>
Epoch 110, loss 0.8792224520712084, correct 50, time 3.88 seconds<br>
Epoch 120, loss 1.0588779945104387, correct 48, time 4.17 seconds<br>
Epoch 130, loss 0.3586506777623135, correct 50, time 3.62 seconds<br>
Epoch 140, loss 0.0033141704232045365, correct 50, time 3.57 seconds<br>
Epoch 150, loss 0.2159235573102668, correct 48, time 4.03 seconds<br>
Epoch 160, loss 1.6465032568216644, correct 48, time 3.48 seconds<br>
Epoch 170, loss 0.7556596802348093, correct 50, time 3.59 seconds<br>
Epoch 180, loss 0.5330359323571452, correct 50, time 3.41 seconds<br>
Epoch 190, loss 0.7515044807146463, correct 49, time 3.41 seconds<br>
Epoch 200, loss 0.8452292745810825, correct 50, time 3.58 seconds<br>
Epoch 210, loss 1.455229025610616, correct 49, time 3.54 seconds<br>
Epoch 220, loss 0.587690827734966, correct 49, time 3.82 seconds<br>
Epoch 230, loss 0.4220511836521446, correct 49, time 3.55 seconds<br>
Epoch 240, loss 1.1336858393192348, correct 50, time 3.44 seconds<br>
Epoch 250, loss 1.3061055945508478, correct 50, time 3.58 seconds<br>
Epoch 260, loss 0.8493427744421964, correct 49, time 3.79 seconds<br>
Epoch 270, loss 0.2277250392912073, correct 50, time 3.50 seconds<br>
Epoch 280, loss 1.3565682322845304, correct 50, time 3.52 seconds<br>
Epoch 290, loss 0.6185873804431764, correct 50, time 3.54 seconds<br>
Epoch 300, loss 0.32445498142476044, correct 50, time 3.66 seconds<br>
Epoch 310, loss 0.5511276251847097, correct 50, time 3.61 seconds<br>
Epoch 320, loss 0.3321710915235155, correct 50, time 3.70 seconds<br>
Epoch 330, loss 0.4487310918252116, correct 50, time 3.78 seconds<br>
Epoch 340, loss 0.2646883675056445, correct 50, time 3.70 seconds<br>
Epoch 350, loss 0.42280397173828356, correct 50, time 3.60 seconds<br>
Epoch 360, loss 0.14173063363225693, correct 50, time 3.69 seconds<br>
Epoch 370, loss 0.4392250927043539, correct 50, time 3.65 seconds<br>
Epoch 380, loss 0.26730525753617935, correct 50, time 3.63 seconds<br>
Epoch 390, loss 0.2908382705646153, correct 50, time 3.78 seconds<br>
Epoch 400, loss 0.015366871381032104, correct 50, time 3.80 seconds<br>
Epoch 410, loss 0.5586355080868729, correct 50, time 3.72 seconds<br>
Epoch 420, loss 0.5977382212745476, correct 50, time 3.58 seconds<br>
Epoch 430, loss 0.3163880578361895, correct 50, time 3.60 seconds<br>
Epoch 440, loss 0.5139085100663011, correct 50, time 3.60 seconds<br>
Epoch 450, loss 0.24089860451886512, correct 50, time 3.62 seconds<br>
Epoch 460, loss 0.07105509053984617, correct 50, time 3.68 seconds<br>
Epoch 470, loss 0.6941177627081114, correct 50, time 3.56 seconds<br>
Epoch 480, loss 0.3962678197074678, correct 50, time 3.59 seconds<br>
Epoch 490, loss 0.38813487147049826, correct 50, time 3.60 seconds<br>

backend: cpu, dataset: split, hidden: 100, rate: 0.05, points: 50<br>
Epoch 0, loss 9.19483399736696, correct 31, time 27.72 seconds<br>
Epoch 10, loss 8.405148817681415, correct 39, time 0.11 seconds<br>
Epoch 20, loss 5.383761765652042, correct 39, time 0.11 seconds<br>
Epoch 30, loss 5.02391445705955, correct 43, time 0.13 seconds<br>
Epoch 40, loss 5.541219299962146, correct 43, time 0.13 seconds<br>
Epoch 50, loss 4.490757296141566, correct 37, time 0.11 seconds<br>
Epoch 60, loss 4.972234363410205, correct 44, time 0.11 seconds<br>
Epoch 70, loss 3.018466397654893, correct 42, time 0.11 seconds<br>
Epoch 80, loss 4.546883788828131, correct 46, time 0.10 seconds<br>
Epoch 90, loss 2.081491967384871, correct 46, time 0.10 seconds<br>
Epoch 100, loss 1.6225524286219792, correct 48, time 0.11 seconds<br>
Epoch 110, loss 2.379710088402255, correct 43, time 0.11 seconds<br>
Epoch 120, loss 1.6126376006614547, correct 47, time 0.10 seconds<br>
Epoch 130, loss 2.4098899788896366, correct 44, time 0.15 seconds<br>
Epoch 140, loss 4.2603655270006415, correct 47, time 0.10 seconds<br>
Epoch 150, loss 1.5852417438696964, correct 47, time 0.11 seconds<br>
Epoch 160, loss 1.9129506162825392, correct 46, time 0.11 seconds<br>
Epoch 170, loss 1.7246702595671435, correct 49, time 0.10 seconds<br>
Epoch 180, loss 2.2362033187498143, correct 47, time 0.11 seconds<br>
Epoch 190, loss 1.6565151370875626, correct 49, time 0.11 seconds<br>
Epoch 200, loss 1.0867448301385187, correct 48, time 0.11 seconds<br>
Epoch 210, loss 2.8660134963232737, correct 46, time 0.12 seconds<br>
Epoch 220, loss 1.7080830178536939, correct 47, time 0.12 seconds<br>
Epoch 230, loss 0.9837844659186151, correct 49, time 0.12 seconds<br>
Epoch 240, loss 1.293236374146267, correct 45, time 0.11 seconds<br>
Epoch 250, loss 1.4869302825292274, correct 46, time 0.12 seconds<br>
Epoch 260, loss 0.8006648695748868, correct 48, time 0.12 seconds<br>
Epoch 270, loss 1.9867720699013034, correct 47, time 0.12 seconds<br>
Epoch 280, loss 2.6384880751299864, correct 49, time 0.11 seconds<br>
Epoch 290, loss 3.4842850867453983, correct 44, time 0.11 seconds<br>
Epoch 300, loss 1.9032442911441771, correct 49, time 0.11 seconds<br>
Epoch 310, loss 1.0205146032244738, correct 49, time 0.11 seconds<br>
Epoch 320, loss 0.2587598025608416, correct 49, time 0.11 seconds<br>
Epoch 330, loss 1.9328023424670397, correct 49, time 0.11 seconds<br>
Epoch 340, loss 0.2778828195423739, correct 49, time 0.11 seconds<br>
Epoch 350, loss 0.5940669571030852, correct 49, time 0.11 seconds<br>
Epoch 360, loss 2.5326698236915206, correct 49, time 0.11 seconds<br>
Epoch 370, loss 0.5411868699100493, correct 49, time 0.13 seconds<br>
Epoch 380, loss 0.9466621037952306, correct 49, time 0.24 seconds<br>
Epoch 390, loss 1.0649361954061662, correct 48, time 0.13 seconds<br>
Epoch 400, loss 0.7954955369284609, correct 49, time 0.14 seconds<br>
Epoch 410, loss 0.7188432472201483, correct 47, time 0.14 seconds<br>
Epoch 420, loss 0.4878156869012766, correct 46, time 0.14 seconds<br>
Epoch 430, loss 3.0469188793542097, correct 47, time 0.14 seconds<br>
Epoch 440, loss 1.3868048337030472, correct 48, time 0.16 seconds<br>
Epoch 450, loss 2.345493699178271, correct 45, time 0.16 seconds<br>
Epoch 460, loss 1.3453385303776253, correct 49, time 0.15 seconds<br>
Epoch 470, loss 0.20964734374049118, correct 50, time 0.14 seconds<br>
Epoch 480, loss 0.1614603977532976, correct 50, time 0.14 seconds<br>
Epoch 490, loss 0.0118887780572651, correct 50, time 0.15 seconds<br>

backend: gpu, dataset: split, hidden: 100, rate: 0.05, points: 50 <br>
Epoch 0, loss 5.657220562848061, correct 32, time 7.19 seconds<br>
Epoch 10, loss 7.771813451817472, correct 20, time 3.59 seconds<br>
Epoch 20, loss 4.390204274814756, correct 40, time 3.49 seconds<br>
Epoch 30, loss 4.682450189826815, correct 45, time 3.60 seconds<br>
Epoch 40, loss 2.4488682144433236, correct 48, time 3.49 seconds<br>
Epoch 50, loss 2.4681724160292617, correct 49, time 3.56 seconds<br>
Epoch 60, loss 2.4261143832587555, correct 45, time 3.70 seconds<br>
Epoch 70, loss 1.7713401043514434, correct 50, time 3.59 seconds<br>
Epoch 80, loss 1.603895997609591, correct 49, time 3.64 seconds<br>
Epoch 90, loss 1.5644928653124834, correct 47, time 3.50 seconds<br>
Epoch 100, loss 2.75301124400299, correct 48, time 3.59 seconds<br>
Epoch 110, loss 1.539822924297964, correct 50, time 3.68 seconds<br>
Epoch 120, loss 1.2809096437610408, correct 49, time 3.61 seconds<br>
Epoch 130, loss 1.022715524899043, correct 50, time 3.56 seconds<br>
Epoch 140, loss 1.5698478627537908, correct 49, time 3.45 seconds<br>
Epoch 150, loss 0.4358144683444658, correct 50, time 3.51 seconds<br>
Epoch 160, loss 0.9595922649696866, correct 50, time 3.46 seconds<br>
Epoch 170, loss 0.5860427930078508, correct 50, time 3.60 seconds<br>
Epoch 180, loss 0.38918018419997474, correct 49, time 3.44 seconds<br>
Epoch 190, loss 0.45877751476919487, correct 50, time 3.52 seconds<br>
Epoch 200, loss 1.3648233947085153, correct 50, time 3.61 seconds<br>
Epoch 210, loss 0.029750519370476115, correct 49, time 3.54 seconds<br>
Epoch 220, loss 0.38641139104088107, correct 49, time 3.64 seconds<br>
Epoch 230, loss 0.613503799054451, correct 50, time 3.69 seconds<br>
Epoch 240, loss 0.3152896433041337, correct 50, time 3.50 seconds<br>
Epoch 250, loss 0.7234473277027575, correct 50, time 3.59 seconds<br>
Epoch 260, loss 0.38205670887584506, correct 50, time 3.62 seconds<br>
Epoch 270, loss 0.676702144328842, correct 50, time 3.51 seconds<br>
Epoch 280, loss 0.18440861828851024, correct 50, time 3.61 seconds<br>
Epoch 290, loss 0.7078128883650542, correct 50, time 3.66 seconds<br>
Epoch 300, loss 0.051181631038723395, correct 50, time 3.59 seconds<br>
Epoch 310, loss 0.5098656692403769, correct 50, time 3.49 seconds<br>
Epoch 320, loss 0.3779500599304156, correct 50, time 3.57 seconds<br>
Epoch 330, loss 0.069764283073878, correct 50, time 3.49 seconds<br>
Epoch 340, loss 0.45750112034239725, correct 50, time 3.55 seconds<br>
Epoch 350, loss 0.3478083485810791, correct 50, time 3.55 seconds<br>
Epoch 360, loss 0.18735739466198425, correct 50, time 3.52 seconds<br>
Epoch 370, loss 0.4813328613380132, correct 50, time 3.61 seconds<br>
Epoch 380, loss 0.2176243847627412, correct 50, time 3.52 seconds<br>
Epoch 390, loss 0.43082906293148514, correct 50, time 3.53 seconds<br>
Epoch 400, loss 0.08128474561533073, correct 50, time 3.56 seconds<br>
Epoch 410, loss 0.0765732570551384, correct 50, time 3.60 seconds<br>
Epoch 420, loss 0.527890284104207, correct 50, time 3.54 seconds<br>
Epoch 430, loss 0.08984507579522029, correct 50, time 3.55 seconds<br>
Epoch 440, loss 0.5046593678901514, correct 50, time 3.64 seconds<br>
Epoch 450, loss 0.04832135654756442, correct 50, time 3.67 seconds<br>
Epoch 460, loss 0.34876685944253306, correct 50, time 3.56 seconds<br>
Epoch 470, loss 0.3345301204433415, correct 50, time 3.67 seconds<br>
Epoch 480, loss 0.0892665537223157, correct 50, time 3.67 seconds<br>
Epoch 490, loss 0.058419291457314515, correct 50, time 3.64 seconds<br>

backend: cpu, dataset: xor, hidden: 100, rate: 0.05, points: 50 <br>
Epoch 0, loss 6.419373459460569, correct 27, time 32.09 seconds <br>
Epoch 10, loss 6.96875208904579, correct 42, time 0.11 seconds <br>
Epoch 20, loss 2.8649128518784948, correct 45, time 0.11 seconds <br>
Epoch 30, loss 3.550069376631923, correct 44, time 0.15 seconds <br>
Epoch 40, loss 3.5270420886898934, correct 46, time 0.11 seconds <br>
Epoch 50, loss 1.4851711271250518, correct 46, time 0.15 seconds <br>
Epoch 60, loss 5.155502500650663, correct 44, time 0.18 seconds <br>
Epoch 70, loss 3.5449007590415262, correct 46, time 0.13 seconds <br>
Epoch 80, loss 2.665217641898387, correct 46, time 0.16 seconds <br>
Epoch 90, loss 1.8992136017299808, correct 47, time 0.14 seconds <br>
Epoch 100, loss 1.304903268343776, correct 46, time 0.12 seconds <br>
Epoch 110, loss 1.7048108392893226, correct 47, time 0.13 seconds <br>
Epoch 120, loss 1.7524713135351644, correct 47, time 0.13 seconds <br>
Epoch 130, loss 0.7800397254325747, correct 47, time 0.14 seconds <br>
Epoch 140, loss 1.3942308970284822, correct 48, time 0.16 seconds <br>
Epoch 150, loss 1.2982502268270872, correct 47, time 0.16 seconds <br>
Epoch 160, loss 2.1938609893261294, correct 47, time 0.15 seconds <br>
Epoch 170, loss 2.0435174243687046, correct 49, time 0.15 seconds <br>
Epoch 180, loss 2.0413721360855894, correct 47, time 0.15 seconds <br>
Epoch 190, loss 0.5956757905091553, correct 46, time 0.11 seconds <br>
Epoch 200, loss 0.36561658763707117, correct 50, time 0.12 seconds <br>
Epoch 210, loss 0.8991228722968455, correct 50, time 0.12 seconds <br>
Epoch 220, loss 1.7723949286574074, correct 50, time 0.13 seconds <br>
Epoch 230, loss 0.29513737368708715, correct 48, time 0.11 seconds <br>
Epoch 240, loss 1.4623761056995532, correct 50, time 0.15 seconds <br>
Epoch 250, loss 0.8397449821010389, correct 48, time 0.15 seconds <br>
Epoch 260, loss 2.6216743801285287, correct 46, time 0.12 seconds <br>
Epoch 270, loss 1.184200650942846, correct 50, time 0.11 seconds <br>
Epoch 280, loss 1.6473918764442097, correct 49, time 0.11 seconds <br>
Epoch 290, loss 0.8798826080569603, correct 50, time 0.12 seconds <br>
Epoch 300, loss 2.5755984258730074, correct 48, time 0.11 seconds <br>
Epoch 310, loss 1.2145702502093394, correct 50, time 0.11 seconds <br>
Epoch 320, loss 0.6204167470151246, correct 50, time 0.14 seconds <br>
Epoch 330, loss 0.7355583584759526, correct 49, time 0.27 seconds <br>
Epoch 340, loss 0.868971228440603, correct 50, time 0.19 seconds <br>
Epoch 350, loss 0.8815149351708758, correct 50, time 0.18 seconds <br>
Epoch 360, loss 1.2464426577660854, correct 50, time 0.19 seconds <br>
Epoch 370, loss 0.6491648972716044, correct 50, time 0.19 seconds <br>
Epoch 380, loss 0.11424263970568796, correct 50, time 0.31 seconds <br>
Epoch 390, loss 0.14650772760751887, correct 49, time 0.19 seconds <br>
Epoch 400, loss 0.8899271349923312, correct 50, time 0.21 seconds <br>
Epoch 410, loss 1.4237794193256832, correct 49, time 0.19 seconds <br>
Epoch 420, loss 0.03991044124805683, correct 50, time 0.20 seconds <br>
Epoch 430, loss 0.55792539173454, correct 50, time 0.15 seconds <br>
Epoch 440, loss 0.3313704762488577, correct 50, time 0.17 seconds <br>
Epoch 450, loss 0.6332956089620122, correct 50, time 0.14 seconds <br>
Epoch 460, loss 0.8425150715231002, correct 50, time 0.20 seconds <br>
Epoch 470, loss 1.2688608887803634, correct 50, time 0.14 seconds <br>
Epoch 480, loss 0.8351903102009907, correct 50, time 0.14 seconds <br>
Epoch 490, loss 0.04955592718754653, correct 50, time 0.14 seconds <br>

backend: gpu, dataset: xor, hidden: 100, rate: 0.05, points: 50 <br>
Epoch 0, loss 7.476310475180543, correct 30, time 7.16 seconds <br>
Epoch 10, loss 4.727476276456226, correct 43, time 3.23 seconds <br>
Epoch 20, loss 4.413027300862569, correct 41, time 3.22 seconds <br>
Epoch 30, loss 3.1480314596256616, correct 44, time 3.21 seconds <br>
Epoch 40, loss 2.6570283062155857, correct 48, time 3.45 seconds <br>
Epoch 50, loss 4.438296014371257, correct 48, time 3.44 seconds <br>
Epoch 60, loss 1.6032287139331776, correct 48, time 3.39 seconds <br>
Epoch 70, loss 1.4258468236981876, correct 47, time 3.16 seconds <br>
Epoch 80, loss 1.288726491884274, correct 47, time 4.18 seconds <br>
Epoch 90, loss 3.2911100403264753, correct 47, time 3.98 seconds <br>
Epoch 100, loss 0.7621658342723722, correct 47, time 3.96 seconds <br>
Epoch 110, loss 1.571021897498789, correct 48, time 4.22 seconds <br>
Epoch 120, loss 0.6844930235280375, correct 49, time 3.69 seconds <br>
Epoch 130, loss 1.8801416915272575, correct 50, time 4.02 seconds <br>
Epoch 140, loss 1.1411761922931292, correct 50, time 3.38 seconds <br>
Epoch 150, loss 2.4027336608925967, correct 47, time 4.15 seconds <br>
Epoch 160, loss 1.7413176130089363, correct 47, time 2.98 seconds <br>
Epoch 170, loss 1.0501054598131099, correct 50, time 3.60 seconds <br>
Epoch 180, loss 1.0586521114711107, correct 50, time 4.06 seconds <br>
Epoch 190, loss 0.8751752813658673, correct 49, time 3.57 seconds <br>
Epoch 200, loss 1.0466373827578237, correct 49, time 3.62 seconds <br>
Epoch 210, loss 0.41247357859702544, correct 50, time 4.36 seconds <br>
Epoch 220, loss 0.5331637899299921, correct 50, time 3.60 seconds <br>
Epoch 230, loss 0.8882712673699491, correct 50, time 4.25 seconds <br>
Epoch 240, loss 0.2177648893900701, correct 50, time 4.47 seconds <br>
Epoch 250, loss 1.2502294932791236, correct 50, time 4.27 seconds <br>
Epoch 260, loss 0.5933495121581995, correct 49, time 4.11 seconds <br>
Epoch 270, loss 0.20401969436413965, correct 50, time 4.23 seconds <br>
Epoch 280, loss 1.0610094414786648, correct 50, time 3.73 seconds <br>
Epoch 290, loss 1.3771229602442395, correct 50, time 4.23 seconds <br>
Epoch 300, loss 0.1131322885396267, correct 50, time 4.23 seconds <br>
Epoch 310, loss 0.013705423278452268, correct 50, time 3.81 seconds <br>
Epoch 320, loss 0.5220105978695037, correct 50, time 3.96 seconds <br>
Epoch 330, loss 0.7794053372675876, correct 50, time 3.77 seconds <br>
Epoch 340, loss 1.6892820268932203, correct 50, time 4.31 seconds <br>
Epoch 350, loss 0.7487402276317382, correct 50, time 3.92 seconds <br>
Epoch 360, loss 1.0347667222290767, correct 50, time 3.83 seconds <br>
Epoch 370, loss 0.9314431572127492, correct 50, time 4.06 seconds <br>
Epoch 380, loss 0.11005200728209993, correct 50, time 4.27 seconds <br>
Epoch 390, loss 0.5068972420186249, correct 50, time 3.91 seconds <br>
Epoch 400, loss 0.2952802977863156, correct 50, time 3.99 seconds <br>
Epoch 410, loss 0.6045660033875919, correct 50, time 3.82 seconds <br>
Epoch 420, loss 0.3167302288069519, correct 50, time 4.22 seconds <br>
Epoch 430, loss 1.0609286614628063, correct 50, time 4.25 seconds <br>
Epoch 440, loss 1.0615912674623067, correct 50, time 3.73 seconds <br>
Epoch 450, loss 0.36235853028028865, correct 50, time 4.10 seconds <br>
Epoch 460, loss 0.8262942080525722, correct 50, time 3.89 seconds <br>
Epoch 470, loss 0.5288081545307892, correct 50, time 3.97 seconds <br>
Epoch 480, loss 1.159035481274209, correct 50, time 4.21 seconds <br>
Epoch 490, loss 1.064632122438207, correct 50, time 3.88 seconds <br>


## Large Model
backend: cpu, dataset: simple, hidden: 250, rate: 0.05, points: 50<br>
Epoch 0, loss 0.7606393531559354, correct 49, time 35.80 seconds<br>
Epoch 10, loss 0.25941216342984647, correct 50, time 0.53 seconds<br>
Epoch 20, loss 0.479705376807187, correct 48, time 0.41 seconds<br>
Epoch 30, loss 0.24701770173009144, correct 50, time 0.40 seconds<br>
Epoch 40, loss 0.29156875549110806, correct 48, time 0.41 seconds<br>
Epoch 50, loss 0.015396037252339529, correct 50, time 0.42 seconds<br>
Epoch 60, loss 0.13417694547685458, correct 50, time 0.42 seconds<br>
Epoch 70, loss 0.22460687213152644, correct 50, time 0.49 seconds<br>
Epoch 80, loss 0.0023624080640160397, correct 50, time 0.59 seconds<br>
Epoch 90, loss 0.5129697161731503, correct 50, time 0.53 seconds<br>
Epoch 100, loss 0.3444883567458985, correct 50, time 0.54 seconds<br>
Epoch 110, loss 0.10833108221723622, correct 50, time 0.64 seconds<br>
Epoch 120, loss 0.040486322647658855, correct 50, time 0.55 seconds<br>
Epoch 130, loss 0.35612188517581767, correct 50, time 0.53 seconds<br>
Epoch 140, loss 0.3062603012960327, correct 50, time 0.58 seconds<br>
Epoch 150, loss 0.19370745330178601, correct 50, time 0.50 seconds<br>
Epoch 160, loss 0.13267991882144273, correct 50, time 0.44 seconds<br>
Epoch 170, loss 0.17216288936913424, correct 50, time 0.42 seconds<br>
Epoch 180, loss 0.01887248989983529, correct 50, time 0.46 seconds<br>
Epoch 190, loss 0.27022015862506515, correct 50, time 0.46 seconds<br>
Epoch 200, loss 0.022726344496494036, correct 50, time 0.40 seconds<br>
Epoch 210, loss 0.22376940465980433, correct 50, time 0.42 seconds<br>
Epoch 220, loss 0.002833265926793888, correct 50, time 0.46 seconds<br>
Epoch 230, loss 0.13874102905349997, correct 50, time 0.54 seconds<br>
Epoch 240, loss 0.00033391868919162797, correct 50, time 0.52 seconds<br>
Epoch 250, loss 0.24121215958688833, correct 50, time 0.52 seconds<br>
Epoch 260, loss 0.008829711518609925, correct 50, time 0.57 seconds<br>
Epoch 270, loss 0.05880268486726747, correct 50, time 0.59 seconds<br>
Epoch 280, loss 0.12900200224119707, correct 50, time 0.58 seconds<br>
Epoch 290, loss 0.13362182892094088, correct 50, time 0.53 seconds<br>
Epoch 300, loss 0.10495298734942801, correct 50, time 0.48 seconds<br>
Epoch 310, loss 0.30206491752603387, correct 50, time 0.51 seconds<br>
Epoch 320, loss 0.0005357039948453944, correct 50, time 0.45 seconds<br>
Epoch 330, loss 0.07813700592831253, correct 50, time 0.44 seconds<br>
Epoch 340, loss 0.13807468597441142, correct 50, time 0.43 seconds<br>
Epoch 350, loss 0.2549917607494568, correct 50, time 0.43 seconds<br>
Epoch 360, loss 0.004676846760506378, correct 50, time 0.43 seconds<br>
Epoch 370, loss 0.005057016980271147, correct 50, time 0.46 seconds<br>
Epoch 380, loss 0.06302449143667904, correct 50, time 0.60 seconds<br>
Epoch 390, loss 0.10731689785920978, correct 50, time 0.49 seconds<br>
Epoch 400, loss 0.04661558585274219, correct 50, time 0.54 seconds<br>
Epoch 410, loss 0.21511055621319977, correct 50, time 0.56 seconds<br>
Epoch 420, loss 0.0684631396731972, correct 50, time 0.50 seconds<br>
Epoch 430, loss 0.003447190149620545, correct 50, time 0.45 seconds<br>
Epoch 440, loss 0.03425895657445207, correct 50, time 0.55 seconds<br>
Epoch 450, loss 0.09722884211891007, correct 50, time 0.44 seconds<br>
Epoch 460, loss 0.000541937981507568, correct 50, time 0.45 seconds<br>
Epoch 470, loss 9.586667429816694e-05, correct 50, time 0.44 seconds<br>
Epoch 480, loss -4.645286439911455e-06, correct 50, time 0.41 seconds<br>
Epoch 490, loss 0.14047485034656124, correct 50, time 0.40 seconds<br>

backend: gpu, dataset: simple, hidden: 250, rate: 0.05, points: 50<br>
Epoch 0, loss 8.949765890057147, correct 44, time 7.30 seconds<br>
Epoch 10, loss 1.1780632152832902, correct 46, time 3.68 seconds<br>
Epoch 20, loss 1.193418335251185, correct 48, time 7.53 seconds<br>
Epoch 30, loss 0.6611865574924014, correct 50, time 7.64 seconds<br>
Epoch 40, loss 1.1058636676993872, correct 48, time 7.68 seconds<br>
Epoch 50, loss 1.4334337108619828, correct 48, time 7.68 seconds<br>
Epoch 60, loss 0.38200889250764447, correct 50, time 8.24 seconds<br>
Epoch 70, loss 0.41254783743161483, correct 50, time 8.72 seconds<br>
Epoch 80, loss 0.45307135126679776, correct 50, time 7.74 seconds<br>
Epoch 90, loss 0.2938271411180099, correct 50, time 8.53 seconds<br>
Epoch 100, loss 0.4660227088237348, correct 50, time 11.32 seconds<br>
Epoch 110, loss 0.174378309327718, correct 50, time 8.70 seconds<br>
Epoch 120, loss 0.04372808799052591, correct 50, time 7.45 seconds<br>
Epoch 130, loss 0.24977199911236714, correct 50, time 9.87 seconds<br>
Epoch 140, loss 0.14940814870846883, correct 50, time 11.10 seconds<br>
Epoch 150, loss 0.15723165075658216, correct 50, time 8.60 seconds<br>
Epoch 160, loss 0.2396529567130677, correct 50, time 7.83 seconds<br>
Epoch 170, loss 0.39917608407227373, correct 50, time 7.59 seconds<br>
Epoch 180, loss 0.1736258876511395, correct 50, time 8.55 seconds<br>
Epoch 190, loss 0.009911098870725533, correct 50, time 8.54 seconds<br>
Epoch 200, loss 0.14869462847652706, correct 50, time 7.91 seconds<br>
Epoch 210, loss 0.052436858503119375, correct 50, time 10.12 seconds<br>
Epoch 220, loss 0.038385657114081415, correct 50, time 8.13 seconds<br>
Epoch 230, loss 0.3084056258684869, correct 50, time 8.36 seconds<br>
Epoch 240, loss 0.2094386972957846, correct 50, time 8.87 seconds<br>
Epoch 250, loss 0.10815639116935303, correct 50, time 7.91 seconds<br>
Epoch 260, loss 0.054297494635497536, correct 50, time 8.07 seconds<br>
Epoch 270, loss 0.23661158920503676, correct 50, time 7.87 seconds<br>
Epoch 280, loss 0.17397329982839102, correct 50, time 7.68 seconds<br>
Epoch 290, loss 0.2787483843355122, correct 50, time 7.50 seconds<br>
Epoch 300, loss 0.3723312825995243, correct 50, time 7.78 seconds<br>
Epoch 310, loss 0.08458010094720228, correct 50, time 7.60 seconds<br>
Epoch 320, loss 0.1364602541713526, correct 50, time 7.60 seconds<br>
Epoch 330, loss 0.28439278559626796, correct 50, time 8.24 seconds<br>
Epoch 340, loss 0.2565198770732753, correct 50, time 7.75 seconds<br>
Epoch 350, loss 0.20623137747835897, correct 50, time 7.89 seconds<br>
Epoch 360, loss 0.27685227407380804, correct 50, time 7.76 seconds<br>
Epoch 370, loss 0.343004522531798, correct 50, time 7.63 seconds<br>
Epoch 380, loss 0.2904001675575363, correct 50, time 7.50 seconds<br>
Epoch 390, loss 0.2896541016013516, correct 50, time 7.89 seconds<br>
Epoch 400, loss 0.3075151855277562, correct 50, time 7.76 seconds<br>
Epoch 410, loss 0.25183282660167956, correct 50, time 7.82 seconds<br>
Epoch 420, loss 0.22602011079889645, correct 50, time 7.81 seconds<br>
Epoch 430, loss 0.1357590995042177, correct 50, time 7.59 seconds<br>
Epoch 440, loss 0.2984249255152432, correct 50, time 7.72 seconds<br>
Epoch 450, loss 0.23805690514704085, correct 50, time 7.73 seconds<br>
Epoch 460, loss 0.056686432035285475, correct 50, time 7.81 seconds<br>
Epoch 470, loss 0.2171512133835172, correct 50, time 7.80 seconds<br>
Epoch 480, loss 0.1042472584587428, correct 50, time 7.90 seconds<br>
Epoch 490, loss 0.1237732953610747, correct 50, time 7.84 seconds<br>