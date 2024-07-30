import os
import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from utils import *
from datasets import dataset_reader
from models import DeepMotionNet
from loss import CustomMultiTaskLoss
import glob


class Trainer(object):
    def __init__(self, options):
        self.options = options
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)

        self.deformed_train_mesh_paths = glob.glob(os.path.join(self.options.dataroot, 'meshes/train/*.vtk'))
        self.deformed_train_img_paths = glob.glob(os.path.join(self.options.dataroot, 'images/train/*.png'))

        self.deformed_validation_mesh_paths = glob.glob(os.path.join(self.options.dataroot, 'meshes/validation/*.vtk'))
        self.deformed_validation_img_paths = glob.glob(os.path.join(self.options.dataroot, 'images/validation/*.png'))

        self.deformed_train_img_dict = get_drr_image_dict(self.deformed_train_img_paths)
        self.deformed_validation_img_dict = get_drr_image_dict(self.deformed_validation_img_paths)

        print('###############Read training set##################')
        self.trained_data_list = dataset_reader(self.options.ref_mesh_path, self.deformed_train_mesh_paths,
                                                self.deformed_train_img_dict, self.options.img_res)
        print('################Read validation set#################')
        self.validation_data_list = dataset_reader(self.options.ref_mesh_path, self.deformed_validation_mesh_paths,
                                                   self.deformed_validation_img_dict, self.options.img_res)

        self.train_loader = DataLoader(dataset=self.trained_data_list, batch_size=self.options.batch_size,
                                       worker_init_fn=seed_worker, shuffle=True)
        self.val_loader = DataLoader(dataset=self.validation_data_list, batch_size=self.options.batch_size,
                                     worker_init_fn=seed_worker, shuffle=False)

        self.model = DeepMotionNet(options).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.options.lr, weight_decay=self.options.wd)
        self.criterion = CustomMultiTaskLoss(options).to(self.device)

        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_latest_checkpoint()

    def train(self):
        self.model.train()

        running_loss = 0.0
        running_l1_loss = 0.0
        running_laplacian_loss = 0.0

        for i_batch, data in enumerate(self.train_loader):
            # Set device options
            data = data.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            y_hat = self.model(data)
            y = data.y.to(self.device)
            input_coords = data.pos.to(self.device)

            multitask_loss, l1_loss, laplacian_loss = self.criterion(y_hat, y, input_coords, data.edge_index)
            multitask_loss.backward()

            self.optimizer.step()

            running_loss += multitask_loss.item() * data.num_graphs
            running_l1_loss += l1_loss.item() * data.num_graphs
            running_laplacian_loss += laplacian_loss.item() * data.num_graphs

        loss = running_loss / len(self.train_loader.dataset)
        loss_l1 = running_l1_loss / len(self.train_loader.dataset)
        loss_laplacian = running_laplacian_loss / len(self.train_loader.dataset)

        return loss, loss_l1, loss_laplacian

    def valid(self):
        self.model.eval()

        running_val_loss = 0.0
        running_l1_loss = 0.0
        running_laplacian_loss = 0.0

        with torch.no_grad():
            for i_batch, data in enumerate(self.val_loader):
                # Set device options
                data = data.to(self.device)
                y_hat = self.model(data)
                y = data.y.to(self.device)
                input_coords = data.pos.to(self.device)

                multitask_loss, l1_loss, laplacian_loss = self.criterion(y_hat, y, input_coords, data.edge_index)

                running_val_loss += multitask_loss.item() * data.num_graphs
                running_l1_loss += l1_loss.item() * data.num_graphs
                running_laplacian_loss += laplacian_loss.item() * data.num_graphs

            val_loss = running_val_loss / len(self.val_loader.dataset)
            loss_l1 = running_l1_loss / len(self.val_loader.dataset)
            loss_laplacian = running_laplacian_loss / len(self.val_loader.dataset)

        return val_loss, loss_l1, loss_laplacian

    def model_train(self):
        best_loss = 100000
        epochs_since_improvement = 0
        start_epoch = 0

        if self.checkpoint is not None:
            start_epoch = self.checkpoint['last_epoch'] + 1
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            best_loss = self.checkpoint['best_loss']
            epochs_since_improvement = self.checkpoint['epochs_since_improvement']

        # Epochs
        for epoch in range(start_epoch, self.options.num_epochs):
            # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
            if epochs_since_improvement == 30:
                break
            if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
                adjust_learning_rate(self.optimizer, 0.8)

            # One epoch's training
            train_loss, train_loss_l1, train_loss_laplacian = self.train()

            # One epoch's validation
            val_loss, val_loss_l1, val_loss_laplacian = self.valid()

            print(
                'Epoch {} of {}, Train Loss: {:.3f}, Validation Loss: {:.3f}, Train L1 Loss: {:.3f}, '
                'Validation L1 Loss: {:.3f}, Train Laplacian Loss: {:.3f}, Validation Laplacian Loss: {:.3f}'.format(
                    epoch, self.options.num_epochs, train_loss, val_loss, train_loss_l1, val_loss_l1, train_loss_laplacian,
                    val_loss_laplacian))

            # Check if there was an improvement
            is_best = val_loss < best_loss
            best_loss = min(best_loss, val_loss)

            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            # Save checkpoint
            self.saver.save_checkpoint(epoch, self.model, self.optimizer, val_loss, best_loss, epochs_since_improvement,
                                       is_best)
