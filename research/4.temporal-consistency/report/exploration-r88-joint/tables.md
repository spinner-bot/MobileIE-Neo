# R88 完整共同 DEV 表

4507部署参数；plain学432、complement学384（rank12时），冻结F0。不是容量/优化等价或零推理开销。

|域|seed|控制|PSNR|Etemp|JOD|
|---|---:|---|---:|---:|---:|
|davis_train_pilot|0|MobileIE_raw|26.475237|10.101737|7.040340|
|davis_train_pilot|0|MobileIE_recursive_self05|26.800044|6.545945|7.423528|
|paired_real_ND_DEV|0|MobileIE_raw|27.252453|3.728773|8.414068|
|paired_real_ND_DEV|0|MobileIE_recursive_self05|27.292047|2.605355|8.373159|
|davis_train_pilot|1|MobileIE_raw|26.456334|9.699145|7.110548|
|davis_train_pilot|1|MobileIE_recursive_self05|26.612977|6.393445|7.408986|
|paired_real_ND_DEV|1|MobileIE_raw|27.004095|3.576601|8.552201|
|paired_real_ND_DEV|1|MobileIE_recursive_self05|26.923483|2.534350|8.512007|
|davis_train_pilot|0|DIS_recursive_self05|27.050821|6.520286|7.447420|
|paired_real_ND_DEV|0|DIS_recursive_self05|27.336567|2.821074|8.400475|
|davis_train_pilot|1|DIS_recursive_self05|26.882562|6.349339|7.440672|
|paired_real_ND_DEV|1|DIS_recursive_self05|27.001759|2.719274|8.535978|
|davis_train_pilot|0|plain_true|26.310937|10.179855|7.011842|
|davis_train_pilot|0|plain_zero|25.852986|10.621784|6.885749|
|davis_train_pilot|0|complement_true|26.306937|10.184845|7.010676|
|davis_train_pilot|0|complement_zero|25.842870|10.632066|6.883451|
|paired_real_ND_DEV|0|plain_true|27.107865|3.863712|8.386009|
|paired_real_ND_DEV|0|plain_zero|26.818357|4.137476|8.336024|
|paired_real_ND_DEV|0|complement_true|27.097944|3.868730|8.384973|
|paired_real_ND_DEV|0|complement_zero|26.772881|4.159391|8.331970|
|davis_train_pilot|1|plain_true|26.285901|9.771766|7.072092|
|davis_train_pilot|1|plain_zero|25.809142|10.221816|6.947038|
|davis_train_pilot|1|complement_true|26.286235|9.777450|7.071317|
|davis_train_pilot|1|complement_zero|25.818785|10.215646|6.951269|
|paired_real_ND_DEV|1|plain_true|26.859191|3.721245|8.522972|
|paired_real_ND_DEV|1|plain_zero|26.499658|4.031465|8.458854|
|paired_real_ND_DEV|1|complement_true|26.854725|3.724431|8.521597|
|paired_real_ND_DEV|1|complement_zero|26.503414|4.028364|8.455350|
