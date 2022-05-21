# Modified from: https://github.com/zhuangdizhu/FedGen.git
import torch
import numpy as np
from utils.GAN_config import RUNCONFIGS, CONFIGS_
import torch.nn.functional as F
import torch.nn as nn
import itertools
from utils.train_utils import trans_cifar10_train, trans_cifar100_train, trans_mnist
from PIL import Image
torch.backends.cudnn.enabled = False

criterion = torch.nn.CrossEntropyLoss().cuda()

def get_dataset_name(dataset):
    dataset=dataset.lower()
    passed_dataset=dataset.lower()
    if 'emnist' in dataset:
        passed_dataset='emnist'
    elif 'mnist' in dataset:
        passed_dataset='mnist'
    elif 'cifar' in dataset:
        passed_dataset='cifar'
    elif 'sent' in dataset:
        passed_dataset = 'sent140'
    else:
        raise ValueError('Unsupported dataset {}'.format(dataset))
    return passed_dataset

def get_log_path(args, algorithm, seed, gan_batch_size=32):
    alg=args.dataset + "_" + algorithm
    alg+="_" + str(args.learning_rate) + "_" + str(args.num_users)
    alg+="u" + "_" + str(args.batch_size) + "b" + "_" + str(args.local_epochs)
    alg=alg + "_" + str(seed)
    if 'fedbkd' in algorithm:
        alg += "_embed" + str(args.embedding)
        if int(gan_batch_size) != int(args.batch_size):
            alg += "_gb" + str(gan_batch_size)
    return alg

def create_generative_model(dataset, algorithm='', model='cnn', embedding=False):
    passed_dataset = dataset
    if 'cnn' in algorithm:
        gen_model = algorithm.split('-')[1]
        passed_dataset+='-' + gen_model
    elif 'fedbkd' in algorithm:
        passed_dataset += '-cnn1'
    return Generator(passed_dataset, model=model, embedding=embedding, latent_layer_idx=-1)

METRICS = ['glob_acc', 'per_acc', 'glob_loss', 'per_loss', 'user_train_time', 'server_agg_time']
MIN_SAMPLES_PER_LABEL=1
GENERATORCONFIGS = {
    'cifar10': (512, 3072, 3, 10, 64),
    'cifar100': (512, 3072, 3, 100, 64),
    'mnist': (256, 32, 1, 10, 32),
    'emnist': (256, 32, 1, 25, 32),
    'femnist': (256, 784, 1, 10, 64),
    'sent140': (512, 7500, 1, 2, 64)
}

class Server:
    def __init__(self, args, seed):

        self.dataset = args.dataset
        self.num_glob_iters = args.num_glob_iters
        self.total_train_samples = 0
        self.users = []
        self.num_users = args.num_users
        self.algorithm = args.algorithm
        self.mode = 'partial' if 'partial' in self.algorithm.lower() else 'all'
        self.seed = seed
        self.deviations = {}
        self.metrics = {key: [] for key in METRICS}
        self.timestamp = None
        self.save_path = ''
        self.args = args
        self.init_ensemble_configs()


    def init_ensemble_configs(self):
        #### used for ensemble learning ####
        dataset_name = get_dataset_name(self.dataset)
        self.ensemble_lr = RUNCONFIGS[dataset_name].get('ensemble_lr', 1e-4)
        self.ensemble_batch_size = RUNCONFIGS[dataset_name].get('ensemble_batch_size', 128)
        self.ensemble_epochs = RUNCONFIGS[dataset_name]['ensemble_epochs']
        self.num_pretrain_iters = RUNCONFIGS[dataset_name]['num_pretrain_iters']
        self.temperature = RUNCONFIGS[dataset_name].get('temperature', 1)
        self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']
        self.ensemble_alpha = RUNCONFIGS[dataset_name].get('ensemble_alpha', 1)
        self.ensemble_beta = RUNCONFIGS[dataset_name].get('ensemble_beta', 0)
        self.ensemble_eta = RUNCONFIGS[dataset_name].get('ensemble_eta', 0.3)
        self.ensemble_gama = RUNCONFIGS[dataset_name].get('ensemble_gama', 0.3)
        self.weight_decay = RUNCONFIGS[dataset_name].get('weight_decay', 0)
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']
        self.ensemble_train_loss = []
        self.n_teacher_iters = self.args.n_teacher_iters
        self.n_student_iters = 1
        print("ensemble_lr: {}".format(self.ensemble_lr))
        print("ensemble_batch_size: {}".format(self.ensemble_batch_size))
        print("unique_labels: {}".format(self.unique_labels))

    def init_loss_fn(self):
        self.loss = nn.NLLLoss()
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))

class Generator(nn.Module):
    def __init__(self, dataset='mnist', model='cnn', embedding=True, latent_layer_idx=-1):
        super(Generator, self).__init__()
        self.embedding = embedding
        self.dataset = dataset
        self.latent_layer_idx = latent_layer_idx
        self.hidden_dim, self.latent_dim, self.input_channel, self.n_class, self.noise_dim = GENERATORCONFIGS[dataset]
        input_dim = self.noise_dim * 2 if self.embedding else self.noise_dim + self.n_class
        self.fc_configs = [input_dim, self.hidden_dim]
        self.init_loss_fn()
        self.build_network()

    def get_number_of_parameters(self):
        pytorch_total_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def init_loss_fn(self):
        self.crossentropy_loss=nn.NLLLoss(reduce=False) # same as above
        self.diversity_loss = DiversityLoss(metric='l1')
        self.dist_loss = nn.MSELoss()


    def build_network(self):
        self.embedding_layer = nn.Embedding(self.n_class, self.noise_dim)
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            print("Build layer {} X {}".format(input_dim, out_dim))
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            self.fc_layers += [fc, bn, act]
        self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim)
        print("Build last layer {} X {}".format(self.fc_configs[-1], self.latent_dim))

    def forward(self, input, dataset='cifar10'):
        if dataset == 'cifar10':
            trans = trans_cifar10_train
        elif dataset == 'cifar100':
            trans = trans_cifar100_train
        elif dataset == 'mnist':
            trans = trans_mnist
        elif dataset == 'femnist':
            trans = None
        elif dataset == 'sent140':
            trans = None
        result = {}
        z = input
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)
        result['output'] = z
        if dataset == 'femnist':
            z = z.reshape((z.shape[0], 1, 28, 28))
            result['img'] = z
        elif dataset == 'sent140':
            z = z.reshape((z.shape[0], 25, 1, 300))
            result['img'] = z
        else:
            z = z.reshape((z.shape[0], 32, 32, 3))
            images = None
            for img in z:
                trans_img = torch.Tensor(trans(Image.fromarray(img.detach().cpu().numpy().astype(np.uint8)))).unsqueeze(
                    dim=0)
                if images is None:
                    images = trans_img
                else:
                    images = torch.cat((images, trans_img), dim=0)
            result['img'] = images
        return result

class Student(nn.Module):
    def __init__(self, args):
        super(Student, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, args.num_classes)
        self.cls = args.num_classes
        self.drop = nn.Dropout(0.6)

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]
        configs, input_channel, self.output_dim, self.hidden_dim, self.latent_dim=CONFIGS_[args.dataset]

    def forward(self, x, logit=False, out_feature=False):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = self.drop(self.fc3(x))
        out=F.log_softmax(z, dim=1)
        if out_feature:
            return out, x
        result = {'output': out}
        if logit:
            result['logit'] = z
        return result

class Student100(nn.Module):
    def __init__(self, args):
        super(Student100, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.6)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, args.num_classes)
        self.cls = args.num_classes

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]
        configs, input_channel, self.output_dim, self.hidden_dim, self.latent_dim=CONFIGS_[args.dataset]

    def forward(self, x, logit=False, out_feature=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.drop((F.relu(self.fc2(x))))
        z = self.fc3(x)
        out=F.log_softmax(z, dim=1)
        if out_feature:
            return out, x
        result = {'output': out}
        if logit:
            result['logit'] = z
        return result

class StudentFemnist(nn.Module):
    def __init__(self, args, dim_in, dim_hidden, dim_out):
        super(StudentFemnist, self).__init__()
        self.layer_input = nn.Linear(dim_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0)
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 64)
        self.layer_out = nn.Linear(64, dim_out)
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x, logit=False, out_feature=False):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x = self.layer_hidden2(x)
        x = self.relu(x)
        z = self.layer_out(x)
        out=F.log_softmax(z, dim=1)
        if out_feature:
            return out, x
        result = {'output': out}
        if logit:
            result['logit'] = z
        return result

class StudentSent(nn.Module):
    def __init__(self, args, dropout=0.5):
        super(StudentSent, self).__init__()
        self.args = args
        self.rnn = getattr(nn, 'LSTM')(25, 128, 1)
        self.fc = nn.Linear(128, 10)
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(10, 2)
        self.nlayers = 1
        self.nhid = 128

    def forward(self, emb, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(1)
        emb = emb.view(300, 1, 25)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(F.relu(self.fc(output)))
        decoded = self.decoder(output[-1, :, :])
        result = {'output': decoded.t()}
        result['hidden'] = hidden
        return result

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

class DF_gan(Server):
    def __init__(self, args, seed, w_locals, selected_users, dataset_test, dict_users_test, user_label_dict, y_input=None, eps=None, label=None):
        super(DF_gan, self).__init__(args, seed)
        self.generative_model = create_generative_model(args.dataset, algorithm=args.algorithm, embedding=args.embedding).to(args.device)
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)
        self.w_locals = w_locals
        self.args = args
        self.selected_users = selected_users
        self.dataset_test = dataset_test
        self.dict_users_test = dict_users_test
        self.user_label_dict = user_label_dict
        self.user_label_unique = []
        self.count_label_unique()
        if y_input is None:
            self.y_input, self.eps, self.label = self.init_y_input()
        else:
            self.y_input, self.eps, self.label = y_input, eps, label

    def init_y_input(self):
        y = np.array(list(itertools.chain.from_iterable([[i]*(self.args.gan_train_nums//self.args.num_classes) for i in range(self.args.num_classes)])))
        np.random.shuffle(y)
        label = torch.LongTensor(y).to(self.args.device)
        batch_size = label.shape[0]
        eps = torch.rand((batch_size, self.generative_model.noise_dim)).to(self.args.device)  # sampling from Gaussian
        y_input = self.generative_model.embedding_layer(label)
        y_input = torch.cat((eps, y_input), dim=1).to(self.args.device)
        return y_input, eps, label

    def count_label_unique(self):
        for user in self.selected_users:
            self.user_label_unique.extend(self.user_label_dict[user])
        self.user_label_unique = list(set(self.user_label_unique))

    def train_generator(self, epoches=5, pre_model=None):
        if pre_model is not None:
            self.generative_model.load_state_dict(pre_model)

        TEACHER_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0

        user_model_list = []
        for user_idx, user in enumerate(self.selected_users):
            if self.args.dataset == 'cifar10':
                user_model = Student(args=self.args).to(self.args.device)
            elif self.args.dataset == 'cifar100':
                user_model = Student100(args=self.args).to(self.args.device)
            elif self.args.dataset == 'femnist':
                user_model = StudentFemnist(args=self.args, dim_in=784, dim_hidden=256, dim_out=self.args.num_classes).to(self.args.device)
            elif self.args.dataset == 'sent140':
                user_model = StudentSent(args=self.args).to(self.args.device)
            user_model.load_state_dict(self.w_locals[user])
            user_model.eval()
            user_model_list.append(user_model)

        def update_generator_(n_iters, TEACHER_LOSS, DIVERSITY_LOSS, tag=False):
            if tag:
                self.user_train_index = {}
            self.generative_model.train()
            for user_idx, user in enumerate(self.selected_users):
                if self.args.dataset == 'sent140':
                    hidden = None
                for i in range(n_iters):
                    loss = 0
                    self.generative_optimizer.zero_grad()
                    user_model = user_model_list[user_idx]
                    usr_label = self.user_label_dict[user]
                    usr_input = None
                    eps_input = None
                    y_label = None
                    for ind, u_label in enumerate(usr_label):
                        label_index = list(itertools.chain.from_iterable(np.argwhere((self.label==u_label).cpu().numpy())))
                        input_index = np.array(label_index)
                        if tag and i == n_iters-1:
                            if user not in self.user_train_index:
                                self.user_train_index[user] = list(input_index)
                            else:
                                self.user_train_index[user].extend(list(input_index))
                        if usr_input is None:
                            usr_input = self.y_input[input_index[0]].unsqueeze(0)
                            eps_input = self.eps[input_index[0]].unsqueeze(0)
                            y_label = self.label[input_index[0]].unsqueeze(0)
                            for i_index in range(1, len(input_index)):
                                usr_input = torch.cat((usr_input, self.y_input[input_index[i_index]].unsqueeze(0)),
                                                      dim=0)
                                eps_input = torch.cat((eps_input, self.eps[input_index[i_index]].unsqueeze(0)),
                                                      dim=0)
                                y_label = torch.cat((y_label, self.label[input_index[0]].unsqueeze(0)), dim=0)
                        else:
                            for i_index in range(len(input_index)):
                                usr_input = torch.cat((usr_input, self.y_input[input_index[i_index]].unsqueeze(0)), dim=0)
                                eps_input = torch.cat((eps_input, self.eps[input_index[i_index]].unsqueeze(0)), dim=0)
                                y_label = torch.cat((y_label, self.label[input_index[0]].unsqueeze(0)), dim=0)

                    gen_result=self.generative_model(usr_input.to(self.args.device), dataset=self.args.dataset)
                    gen_output, gen_img =gen_result['output'].to(self.args.device), gen_result['img'].to(self.args.device)
                    # diversity loss
                    diversity_loss = self.generative_model.diversity_loss(eps_input, gen_output)  # encourage different outputs
                    ######### get teacher loss ############
                    if self.args.dataset == 'sent140':
                        teacher_loss = 0
                        for ind,g_img in enumerate(gen_img):
                            user_result_given_gen = user_model(g_img.detach(), hidden=hidden)
                            hidden = user_result_given_gen['hidden']
                            output = user_result_given_gen['output'].reshape(-1).unsqueeze(dim=0)
                            pred = output.data.max(1)[1]
                            teacher_loss += criterion(output, pred).mean()
                        teacher_loss = teacher_loss / len(gen_img)
                    else:
                        user_result_given_gen = user_model(gen_img, logit=True)
                        pred = user_result_given_gen['logit'].data.max(1)[1]
                        teacher_loss = criterion(user_result_given_gen['logit'], pred).mean()
                    loss += self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss
                    loss.backward(retain_graph=True)
                    self.generative_optimizer.step()
            return TEACHER_LOSS, DIVERSITY_LOSS

        for i in range(epoches):
            if i == epoches - 1:
                tag = True
            else:
                tag = False
            TEACHER_LOSS, DIVERSITY_LOSS=update_generator_(
                self.n_teacher_iters, TEACHER_LOSS, DIVERSITY_LOSS, tag)
        self.generative_lr_scheduler.step()

    def generate_data(self):
        y_input = None
        for ind in range(len(self.y_input)):
            if y_input is None:
                y_input = self.y_input[ind].unsqueeze(0)
            else:
                y_input = torch.cat((y_input, self.y_input[ind].unsqueeze(0)), dim=0)
        data = self.generative_model(y_input, dataset=self.args.dataset)['img']
        dataset = {}
        dataset['x'] = data.detach()
        dataset['y'] = y_input
        return dataset, self.user_train_index
