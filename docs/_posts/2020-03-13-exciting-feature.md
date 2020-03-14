---
title: "Exciting New Feature!"
date:   2020-03-13
excerpt: "Here is an excerpt about our new feature!"
---

Feature!!

# Next Section
Check out this nice code:
```python
# loop to deal with groups
for i, param_group in enumerate(self.optimizer.param_groups):
    # push this group to list before modify
    self.fp16_groups.append(param_group['params'])

    self.fp16_groups_flat.append(
        flatten_dense_tensors_aligned(
            self.fp16_groups[i],
            dist.get_world_size(group=self.dp_process_group),
            self.dp_process_group))

    # set model fp16 weight to slices of flattened buffer
    updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i],
                                              self.fp16_groups[i])
    for p, q in zip(self.fp16_groups[i], updated_params):
        p.data = q.data
```
