# multimodal_sim_interp_qe
Experiments with multimodal quality estimation for simultaneous interpretation.

# Dependencies

numpy (https://www.numpy.org)

xgboost (https://github.com/dmlc/xgboost)

# Example usage

```[bash]
python train.py --joint_training --text_features --interp_audio --src_audio --tuned
```
or
```[bash]
python train.py -jt -tf -ia -sa -tu
```
