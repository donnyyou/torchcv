import os
import json
import argparse


def trans_gt(protocols_dir, new_protocols_dir):
    for tag in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'dev']:
        train_vis_file = 'vis_train_{}.txt'.format(tag)
        train_nir_file = 'nir_train_{}.txt'.format(tag)
        val_vis_file = 'vis_gallery_{}.txt'.format(tag)
        val_nir_file = 'nir_probe_{}.txt'.format(tag)

        label_id = 0
        train_label_dict = dict()
        trainA_json_list = list()
        trainB_json_list = list()
        valA_json_list = list()
        valB_json_list = list()
        with open(os.path.join(protocols_dir, train_nir_file), 'r') as fr:
            for line in fr.readlines():
                name = line.strip().split('\\')[-2]
                if name not in train_label_dict:
                    train_label_dict[name] = label_id
                    label_id += 1

                trainA_json_list.append(
                    dict(image_path=line.strip().replace('\\', '/'), label=train_label_dict[name], name=name)
                )

        with open(os.path.join(protocols_dir, train_vis_file), 'r') as fr:
            for line in fr.readlines():
                name = line.strip().split('\\')[-2]
                assert name in train_label_dict
                trainB_json_list.append(
                    dict(image_path=line.strip().replace('\\', '/'), label=train_label_dict[name], name=name)
                )

        label_id = 0
        val_label_dict = dict()
        with open(os.path.join(protocols_dir, val_nir_file), 'r') as fr:
            for line in fr.readlines():
                name = line.strip().split('\\')[-2]
                if name not in val_label_dict:
                    val_label_dict[name] = label_id
                    label_id += 1

                valA_json_list.append(
                    dict(image_path=line.strip().replace('\\', '/'), label=val_label_dict[name], name=name)
                )

        with open(os.path.join(protocols_dir, val_vis_file), 'r') as fr:
            for line in fr.readlines():
                name = line.strip().split('\\')[-2]
                assert name in val_label_dict
                valB_json_list.append(
                    dict(image_path=line.strip().replace('\\', '/'), label=val_label_dict[name], name=name)
                )

        fw = open(os.path.join(new_protocols_dir, 'train_label{}A.json'.format(tag)), 'w')
        fw.write(json.dumps(trainA_json_list))
        fw.close()
        fw = open(os.path.join(new_protocols_dir, 'train_label{}B.json'.format(tag)), 'w')
        fw.write(json.dumps(trainB_json_list))
        fw.close()
        fw = open(os.path.join(new_protocols_dir, 'val_label{}A.json'.format(tag)), 'w')
        fw.write(json.dumps(valA_json_list))
        fw.close()
        fw = open(os.path.join(new_protocols_dir, 'val_label{}B.json'.format(tag)), 'w')
        fw.write(json.dumps(valB_json_list))
        fw.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--protocols_dir', default=None, type=str,
                        dest='protocols_dir', help='The directory of protocols.')
    parser.add_argument('--new_protocols_dir', default=None, type=str,
                        dest='new_protocols_dir', help='The directory of protocols.')
    args = parser.parse_args()
    if not os.path.exists(args.new_protocols_dir):
        os.makedirs(args.new_protocols_dir)

    trans_gt(args.protocols_dir, args.new_protocols_dir)
