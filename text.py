def generate_data_description(save_dir, reorder):
    """
    create a dataset description file, which consists of images, labels
    """
    # pa100k_data = loadmat('/mnt/data1/jiajian/dataset/attribute/PA100k/annotation.mat')
    pa100k_data = loadmat(os.path.join(save_dir, 'annotation.mat'))

    dataset = EasyDict()
    dataset.description = 'pa100k'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'data')

    train_image_name = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
    val_image_name = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
    test_image_name = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
    all_image_name = train_image_name + val_image_name + test_image_name

    # 仅选择前2000张照片
    if len(all_image_name) > 2000:
        all_image_name = all_image_name[:2000]

    dataset.image_name = all_image_name

    dataset.label = np.concatenate((pa100k_data['train_label'], pa100k_data['val_label'], pa100k_data['test_label']), axis=0)
    # 相应地截取标签数据
    if len(dataset.label) > 2000:
        dataset.label = dataset.label[:2000]

    dataset.attr_name = [pa100k_data['attributes'][i][0][0] for i in range(26)]

    dataset.label_idx = EasyDict()
    dataset.label_idx.eval = list(range(26))

    if reorder:
        dataset.label_idx.eval = group_order

    # 修改数据集划分，仅包含前2000张照片的索引
    dataset.partition = EasyDict()
    dataset.partition.train = np.arange(0, min(2000, 80000))
    dataset.partition.val = np.arange(0, 0)  # 若只使用前2000张训练，验证集和测试集可设为空
    dataset.partition.test = np.arange(0, 0)
    dataset.partition.trainval = np.arange(0, min(2000, 80000))

    dataset.weight_train = np.mean(dataset.label[dataset.partition.train], axis=0).astype(np.float32)
    dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)

    with open(os.path.join(save_dir, 'dataset_all.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)