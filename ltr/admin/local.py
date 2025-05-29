class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/lab/share/share_ssd/wangjh/vision_track/code/FAEMTtrack/ltr'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.pregenerated_masks = ''
        self.lasot_dir = '/home/lab/share/share_3/wangjh/dataset/lasot'
        self.got10k_dir = '/home/lab/share/share_3/wangjh/dataset/got10k/full_data/train'
        self.trackingnet_dir = '/home/lab/share/share_3/wangjh/dataset/trackingnet'
        self.coco_dir = '/home/lab/share/share_3/wangjh/dataset/coco'
        self.mdot_dir = '/home/lab/share/share_3/wangjh/dataset/MDOT'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.lasot_candidate_matching_dataset_path = ''
