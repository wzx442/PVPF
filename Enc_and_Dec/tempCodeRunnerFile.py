# for i in range(num_users):
#     # 先将原始参数赋值给加密参数作为初始值
#     encrypted_params[i] = params[i].copy()
    
#     # 对每个用户j进行处理
#     for j in range(num_users):
#         if i == j:
#             continue
#         # 使用seed_pairs[i,j]作为随机数生成器的种子，加密规则为 encrypted_param[i] = encrypted_param[i] + \\sum_{i < j} PRG(seed_pairs[i,j]) - \\sum_{i > j} PRG(seed_pairs[i,j]) 。PRG(seed_pairs[i,j])是长度为len(encrypted_param)的随机数序列
#         param_length = len(params[i])
#         if i < j:
#             seed = hash((seed_pairs[i,j]))
#             np.random.seed(seed)
#             random_values = np.random.rand(param_length)
#             for idx in range(param_length):
#                 encrypted_params[i][idx] += random_values[idx]  # 累加噪声
#         elif i > j:
#             seed = hash((seed_pairs[j,i]))
#             np.random.seed(seed)
#             random_values = np.random.rand(param_length)
#             for idx in range(param_length):
#                 encrypted_params[i][idx] -= random_values[idx]  # 累减噪声