import torch
import torch.nn.functional as F

def unc_inference(model, dataloader, config, rank=0, world_size=1):
    """
    DONE
    Run multiple inference and obtain uncertainty results for each point
    And for each scene:
    - save the statistics of the uncertainty in obj file (torch save)
    - save the predicted labelds in obj file (torch save)

    required config:
    - unc_round
    - unc_result_dir
    - test_batch_size

    call unc_render to obtain ply files
    """

    # Inference should be done on one GPU
    if world_size > 1 and rank > 0:
        return
    device = rank

    # Set model to eval mode except for the last dropout layer
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            print(f"Warning: module {m.__class__.__name__} set to train")
            m.train()

    # Read scene names
    # Scenes from training datasets
    # with open('splits/scannet/scannetv2_train.txt') as f:
    #     names = sorted([i.strip() for i in f.readlines()])

    with torch.no_grad():
        scene_cnt = 0
        for step, batched_data in enumerate(dataloader):
            print(f"Info: {step}/{len(dataloader)} done")

            # Load batched data
            batched_coords, batched_feats, _ = batched_data["teacher"]
            # batched_feats[:, :3] = batched_feats[:, :3] / 255. - 0.5
            batch_size = len(batched_coords)

            multiround_batched_scores = []

            # Feed forward for multiple rounds
            # batched_sparse_input = ME.SparseTensor(batched_feats.to(device), batched_coords.to(device))
            for i in range(config.unc_round):
                print(f"Info: ---> feed forward #{i} round")
                output_voxel, _ = model(batched_feats, batched_coords, batch_size)
                outputs = []
                for i in range(batch_size):
                    outputs.append(output_voxel[i, :, batched_coords[i][:, 0], batched_coords[i][:, 1], batched_coords[i][:, 2]])
                
                outputs = torch.cat(outputs, dim=1).T
                batched_scores = F.softmax(outputs, dim=1)  # + IMPORTANT STEPS
                multiround_batched_scores.append(batched_scores)

            # batch_size * point_count * class_count
            multiround_batched_scores = torch.stack(multiround_batched_scores)

            batched_labels = multiround_batched_scores.sum(dim=0).argmax(dim=1)  # Label
            batched_scores = multiround_batched_scores.var(dim=0)  # Uncertainty

            # Save labels and uncertainty for each scene
            for scene_id in range(config["test_batch_size"]):
                print(f"Info: ---> processing #{scene_id} scene in the batch")
                selector = batched_coords[:, 0] == scene_id
                single_scene_labels = batched_labels[selector]
                single_scene_scores = batched_scores[selector]

                torch.save(single_scene_labels.cpu(), f"{config.unc_result_dir}/{names[scene_cnt].split('.')[0]}_predicted.obj")
                torch.save(single_scene_scores.cpu(), f"{config.unc_result_dir}/{names[scene_cnt].split('.')[0]}_unc.obj")

                scene_cnt += 1
