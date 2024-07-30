import os
import torch


class CheckpointSaver(object):
    """Class that handles saving and loading checkpoints during training."""

    def __init__(self, save_dir):
        self.save_dir = os.path.abspath(save_dir)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.latest_checkpoint = self.get_latest_checkpoint()
        self.best_checkpoint = self.get_best_checkpoint()

    def exists_checkpoint(self):
        """Check if a checkpoint exists in the current directory."""
        status = False if self.latest_checkpoint is None else True
        return status

    def exists_best_checkpoint(self):
        """Check if a checkpoint exists in the current directory."""
        status = False if self.best_checkpoint is None else True
        return status

    def save_checkpoint(self, epoch, model, optimizer, val_loss, best_loss, epochs_since_improvement, is_best):
        state = {'last_epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'val_loss': val_loss,
                 'best_loss': best_loss,
                 'epochs_since_improvement': epochs_since_improvement}
        filename = '{}/checkpoint.tar'.format(self.save_dir)
        torch.save(state, filename)
        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
        if is_best:
            torch.save(state, '{}/BEST_checkpoint.tar'.format(self.save_dir))

    def load_latest_checkpoint(self):
        """Load a checkpoint."""
        checkpoint = torch.load(self.latest_checkpoint)

        return checkpoint

    def load_best_check_point(self):
        return torch.load(self.best_checkpoint)

    def get_latest_checkpoint(self):
        """Get filename of latest checkpoint if it exists."""
        checkpoint_save_path = os.path.abspath(os.path.join(self.save_dir, 'checkpoint.tar'))
        latest_checkpoint = checkpoint_save_path if (os.path.exists(checkpoint_save_path)) else None

        return latest_checkpoint

    def get_best_checkpoint(self):
        best_check_point_save_path = os.path.abspath(os.path.join(self.save_dir, 'BEST_checkpoint.tar'))
        best_checkpoint = best_check_point_save_path if (os.path.exists(best_check_point_save_path)) else None

        return best_checkpoint




