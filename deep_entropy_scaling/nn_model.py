import torch
import torch.nn as nn


class model0_a(torch.nn.Module):

    def __init__(self, embedding_size, entropy_feature_size, ref_feature_size):
        super(model0_a, self).__init__()

        self.embedding_size = embedding_size

        self.entropy_feature_size = entropy_feature_size
        self.entropy_input_size = embedding_size
        self.ref_feature_size = ref_feature_size
        self.ref_input_size = embedding_size
        return

    def build_entropy_feature_net(self, inner_layer_units, batch_norm=False):
        """
        entropy_feature_net gets entropy features from entropy

        x_in = [ entropy ]
        """

        layer_units = [1] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size]
        self.entropy_feature_net_layer_units = layer_units
        self.entropy_feature_net_batch_norm = batch_norm

        entropy_feature_net_modules = []
        Layers = self.entropy_feature_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            entropy_feature_net_modules.append(layer)
            if i < L-2:
                if self.entropy_feature_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    entropy_feature_net_modules.append(layer)
                print("relu")
                entropy_feature_net_modules.append(torch.nn.ReLU())
        self.entropy_feature_net = nn.Sequential(*entropy_feature_net_modules)
        # else:
        #     self.entropy_feature_net = self.get_polynomial_features
        return

    def build_ref_feature_net(self, inner_layer_units, batch_norm=False):
        """
        ref_feature_net gets entropy features from entropy

        x_in = [ temperature ]
        """

        layer_units = [1] + list(inner_layer_units) + [self.ref_feature_size]
        self.ref_feature_net_layer_units = layer_units
        self.ref_feature_net_batch_norm = batch_norm

        ref_feature_net_modules = []
        Layers = self.ref_feature_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_feature_net_modules.append(layer)
            if i < L-2:
                if self.ref_feature_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_feature_net_modules.append(layer)
                print("relu")
                ref_feature_net_modules.append(torch.nn.ReLU())
        self.ref_feature_net = nn.Sequential(*ref_feature_net_modules)
        return

    def build_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """

        layer_units = [self.entropy_input_size] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size]
        self.parameter_net_layer_units = layer_units
        self.parameter_net_batch_norm = batch_norm

        parameter_net_modules = []
        Layers = self.parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            parameter_net_modules.append(layer)
            if i < L-2:
                if self.parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    parameter_net_modules.append(layer)
                print("relu")
                parameter_net_modules.append(torch.nn.ReLU())
        self.parameter_net = nn.Sequential(*parameter_net_modules)

        return

    def build_ref_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        ref_parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """
        layer_units = [self.ref_input_size] + list(inner_layer_units)
        layer_units += [self.ref_feature_size]
        self.ref_parameter_net_layer_units = layer_units
        self.ref_parameter_net_batch_norm = batch_norm

        ref_parameter_net_modules = []
        Layers = self.ref_parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_parameter_net_modules.append(layer)
            if i < L-2:
                if self.ref_parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_parameter_net_modules.append(layer)
                print("relu")
                ref_parameter_net_modules.append(torch.nn.ReLU())
        self.ref_parameter_net = nn.Sequential(*ref_parameter_net_modules)

        return

    def forward_ref(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        x_ref = self.ref_feature_net(temperature)
        p_ref = self.ref_parameter_net(embedding)
        return torch.sum(p_ref*x_ref, dim=1)
    
    def forward(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        x_ref = self.ref_feature_net(temperature)
        x_entropy = self.entropy_feature_net(entropy)

        # two independent parameter nets
        p_ref = self.ref_parameter_net(embedding)
        p_entropy = self.parameter_net(embedding)

        v_entropy = torch.sum(p_entropy*x_entropy, dim=1)
        v_ref = torch.sum(p_ref*x_ref, dim=1)
        return torch.unsqueeze(v_entropy + v_ref, 1)


class model0_ci(torch.nn.Module):

    def __init__(self, embedding_size, entropy_feature_size, ref_feature_size):
        super(model0_ci, self).__init__()

        self.embedding_size = embedding_size

        self.entropy_feature_size = entropy_feature_size
        self.entropy_input_size = embedding_size + ref_feature_size
        self.ref_feature_size = ref_feature_size
        self.ref_input_size = embedding_size
        return

    def build_entropy_feature_net(self, inner_layer_units, batch_norm=False):
        """
        entropy_feature_net gets entropy features from entropy

        x_in = [ entropy ]
        """

        layer_units = [1] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size]
        self.entropy_feature_net_layer_units = layer_units
        self.entropy_feature_net_batch_norm = batch_norm

        entropy_feature_net_modules = []
        Layers = self.entropy_feature_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            entropy_feature_net_modules.append(layer)
            if i < L-2:
                if self.entropy_feature_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    entropy_feature_net_modules.append(layer)
                print("relu")
                entropy_feature_net_modules.append(torch.nn.ReLU())
        self.entropy_feature_net = nn.Sequential(*entropy_feature_net_modules)
        # else:
        #     self.entropy_feature_net = self.get_polynomial_features
        return

    def build_ref_feature_net(self, inner_layer_units, batch_norm=False):
        """
        ref_feature_net gets entropy features from entropy

        x_in = [ temperature ]
        """

        layer_units = [1] + list(inner_layer_units) + [self.ref_feature_size]
        self.ref_feature_net_layer_units = layer_units
        self.ref_feature_net_batch_norm = batch_norm

        ref_feature_net_modules = []
        Layers = self.ref_feature_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_feature_net_modules.append(layer)
            if i < L-2:
                if self.ref_feature_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_feature_net_modules.append(layer)
                print("relu")
                ref_feature_net_modules.append(torch.nn.ReLU())
        self.ref_feature_net = nn.Sequential(*ref_feature_net_modules)
        return

    def build_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """

        layer_units = [self.entropy_input_size] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size]
        self.parameter_net_layer_units = layer_units
        self.parameter_net_batch_norm = batch_norm

        parameter_net_modules = []
        Layers = self.parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            parameter_net_modules.append(layer)
            if i < L-2:
                if self.parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    parameter_net_modules.append(layer)
                print("relu")
                parameter_net_modules.append(torch.nn.ReLU())
        self.parameter_net = nn.Sequential(*parameter_net_modules)

        return

    def build_ref_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        ref_parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """
        layer_units = [self.ref_input_size] + list(inner_layer_units)
        layer_units += [self.ref_feature_size]
        self.ref_parameter_net_layer_units = layer_units
        self.ref_parameter_net_batch_norm = batch_norm

        ref_parameter_net_modules = []
        Layers = self.ref_parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_parameter_net_modules.append(layer)
            if i < L-2:
                if self.ref_parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_parameter_net_modules.append(layer)
                print("relu")
                ref_parameter_net_modules.append(torch.nn.ReLU())
        self.ref_parameter_net = nn.Sequential(*ref_parameter_net_modules)

        return

    def forward_ref(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        x_ref = self.ref_feature_net(temperature)
        p_ref = self.ref_parameter_net(embedding)
        return torch.sum(p_ref*x_ref, dim=1)
    
    def forward(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        x_ref = self.ref_feature_net(temperature)
        x_entropy = self.entropy_feature_net(entropy)

        # two independent parameter nets
        p_ref = self.ref_parameter_net(embedding)
        p_in = torch.cat([embedding, p_ref], 1)
        p_entropy = self.parameter_net(p_in)

        v_entropy = torch.sum(p_entropy*x_entropy, dim=1)
        v_ref = torch.sum(p_ref*x_ref, dim=1)
        return torch.unsqueeze(v_entropy + v_ref, 1)

class model0_ci_norm(torch.nn.Module):

    def __init__(self, embedding_size, entropy_feature_size, ref_feature_size):
        super(model0_ci_norm, self).__init__()

        self.embedding_size = embedding_size

        self.entropy_feature_size = entropy_feature_size
        self.entropy_input_size = embedding_size + ref_feature_size
        self.ref_feature_size = ref_feature_size
        self.ref_input_size = embedding_size
        return

    def build_entropy_feature_net(self, inner_layer_units, batch_norm=False):
        """
        entropy_feature_net gets entropy features from entropy

        x_in = [ entropy ]
        """

        layer_units = [1] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size]
        self.entropy_feature_net_layer_units = layer_units
        self.entropy_feature_net_batch_norm = batch_norm

        entropy_feature_net_modules = []
        Layers = self.entropy_feature_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            entropy_feature_net_modules.append(layer)
            if i < L-2:
                if self.entropy_feature_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    entropy_feature_net_modules.append(layer)
                print("relu")
                entropy_feature_net_modules.append(torch.nn.ReLU())
        self.entropy_feature_net = nn.Sequential(*entropy_feature_net_modules)
        # else:
        #     self.entropy_feature_net = self.get_polynomial_features
        return

    def build_ref_feature_net(self, inner_layer_units, batch_norm=False):
        """
        ref_feature_net gets entropy features from entropy

        x_in = [ temperature ]
        """

        layer_units = [1] + list(inner_layer_units) + [self.ref_feature_size]
        self.ref_feature_net_layer_units = layer_units
        self.ref_feature_net_batch_norm = batch_norm

        ref_feature_net_modules = []
        Layers = self.ref_feature_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_feature_net_modules.append(layer)
            if i < L-2:
                if self.ref_feature_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_feature_net_modules.append(layer)
                print("relu")
                ref_feature_net_modules.append(torch.nn.ReLU())
        self.ref_feature_net = nn.Sequential(*ref_feature_net_modules)
        return

    def build_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """

        layer_units = [self.entropy_input_size] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size]
        self.parameter_net_layer_units = layer_units
        self.parameter_net_batch_norm = batch_norm

        parameter_net_modules = []
        Layers = self.parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            parameter_net_modules.append(layer)
            if i < L-2:
                if self.parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    parameter_net_modules.append(layer)
                print("relu")
                parameter_net_modules.append(torch.nn.ReLU())
        self.parameter_net = nn.Sequential(*parameter_net_modules)

        return

    def build_ref_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        ref_parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """
        layer_units = [self.ref_input_size] + list(inner_layer_units)
        layer_units += [self.ref_feature_size+1]
        self.ref_parameter_net_layer_units = layer_units
        self.ref_parameter_net_batch_norm = batch_norm

        ref_parameter_net_modules = []
        Layers = self.ref_parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_parameter_net_modules.append(layer)
            if i < L-2:
                if self.ref_parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_parameter_net_modules.append(layer)
                print("relu")
                ref_parameter_net_modules.append(torch.nn.ReLU())
        self.ref_parameter_net = nn.Sequential(*ref_parameter_net_modules)

        return

    def forward_ref(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        dummy = self.ref_parameter_net(embedding)
        p_ref, temperature0 = torch.split(dummy, [self.ref_feature_size,1], 1)
        x_ref = self.ref_feature_net(temperature/temperature0)
        return torch.sum(p_ref*x_ref, dim=1)
    
    def forward(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        dummy = self.ref_parameter_net(embedding)
        p_ref, temperature0 = torch.split(dummy, [self.ref_feature_size,1], 1)
        x_ref = self.ref_feature_net(temperature/temperature0)

        # two independent parameter nets
        p_in = torch.cat([embedding, p_ref], 1)
        p_entropy = self.parameter_net(p_in)
        x_entropy = self.entropy_feature_net(entropy)

        v_entropy = torch.sum(p_entropy*x_entropy, dim=1)
        v_ref = torch.sum(p_ref*x_ref, dim=1)
        return torch.unsqueeze(v_entropy + v_ref, 1)

class model0_ci_enorm(torch.nn.Module):

    def __init__(self, embedding_size, entropy_feature_size, ref_feature_size):
        super(model0_ci_enorm, self).__init__()

        self.embedding_size = embedding_size

        self.entropy_feature_size = entropy_feature_size
        self.entropy_input_size = embedding_size + ref_feature_size
        self.ref_feature_size = ref_feature_size
        self.ref_input_size = embedding_size
        return

    def build_entropy_feature_net(self, inner_layer_units, batch_norm=False):
        """
        entropy_feature_net gets entropy features from entropy

        x_in = [ entropy ]
        """

        layer_units = [1] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size]
        self.entropy_feature_net_layer_units = layer_units
        self.entropy_feature_net_batch_norm = batch_norm

        entropy_feature_net_modules = []
        Layers = self.entropy_feature_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            entropy_feature_net_modules.append(layer)
            if i < L-2:
                if self.entropy_feature_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    entropy_feature_net_modules.append(layer)
                print("relu")
                entropy_feature_net_modules.append(torch.nn.ReLU())
        self.entropy_feature_net = nn.Sequential(*entropy_feature_net_modules)
        # else:
        #     self.entropy_feature_net = self.get_polynomial_features
        return

    def build_ref_feature_net(self, inner_layer_units, batch_norm=False):
        """
        ref_feature_net gets entropy features from entropy

        x_in = [ temperature ]
        """

        layer_units = [1] + list(inner_layer_units) + [self.ref_feature_size]
        self.ref_feature_net_layer_units = layer_units
        self.ref_feature_net_batch_norm = batch_norm

        ref_feature_net_modules = []
        Layers = self.ref_feature_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_feature_net_modules.append(layer)
            if i < L-2:
                if self.ref_feature_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_feature_net_modules.append(layer)
                print("relu")
                ref_feature_net_modules.append(torch.nn.ReLU())
        self.ref_feature_net = nn.Sequential(*ref_feature_net_modules)
        return

    def build_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """

        layer_units = [self.entropy_input_size] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size+1]
        self.parameter_net_layer_units = layer_units
        self.parameter_net_batch_norm = batch_norm

        parameter_net_modules = []
        Layers = self.parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            parameter_net_modules.append(layer)
            if i < L-2:
                if self.parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    parameter_net_modules.append(layer)
                print("relu")
                parameter_net_modules.append(torch.nn.ReLU())
        self.parameter_net = nn.Sequential(*parameter_net_modules)

        return

    def build_ref_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        ref_parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """
        layer_units = [self.ref_input_size] + list(inner_layer_units)
        layer_units += [self.ref_feature_size]
        self.ref_parameter_net_layer_units = layer_units
        self.ref_parameter_net_batch_norm = batch_norm

        ref_parameter_net_modules = []
        Layers = self.ref_parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_parameter_net_modules.append(layer)
            if i < L-2:
                if self.ref_parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_parameter_net_modules.append(layer)
                print("relu")
                ref_parameter_net_modules.append(torch.nn.ReLU())
        self.ref_parameter_net = nn.Sequential(*ref_parameter_net_modules)

        return

    def forward_ref(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        p_ref = self.ref_parameter_net(embedding)
        x_ref = self.ref_feature_net(temperature)
        return torch.sum(p_ref*x_ref, dim=1)
    
    def forward(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        p_ref = self.ref_parameter_net(embedding)
        x_ref = self.ref_feature_net(temperature)

        # two independent parameter nets
        p_in = torch.cat([embedding, p_ref], 1)
        dummy = self.parameter_net(p_in)
        p_entropy, entropy0 = torch.split(dummy, [self.entropy_feature_size,1], 1)
        x_entropy = self.entropy_feature_net(entropy/entropy0)

        v_entropy = torch.sum(p_entropy*x_entropy, dim=1)
        v_ref = torch.sum(p_ref*x_ref, dim=1)
        return torch.unsqueeze(v_entropy + v_ref, 1)


class model0_ci_norm2(torch.nn.Module):

    def __init__(self, embedding_size, entropy_feature_size, ref_feature_size):
        super(model0_ci_norm2, self).__init__()

        self.embedding_size = embedding_size

        self.entropy_feature_size = entropy_feature_size
        self.entropy_input_size = embedding_size + ref_feature_size
        self.ref_feature_size = ref_feature_size
        self.ref_input_size = embedding_size
        return

    def build_entropy_feature_net(self, inner_layer_units, batch_norm=False):
        """
        entropy_feature_net gets entropy features from entropy

        x_in = [ entropy ]
        """

        layer_units = [1] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size]
        self.entropy_feature_net_layer_units = layer_units
        self.entropy_feature_net_batch_norm = batch_norm

        entropy_feature_net_modules = []
        Layers = self.entropy_feature_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            entropy_feature_net_modules.append(layer)
            if i < L-2:
                if self.entropy_feature_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    entropy_feature_net_modules.append(layer)
                print("relu")
                entropy_feature_net_modules.append(torch.nn.ReLU())
        self.entropy_feature_net = nn.Sequential(*entropy_feature_net_modules)
        # else:
        #     self.entropy_feature_net = self.get_polynomial_features
        return

    def build_ref_feature_net(self, inner_layer_units, batch_norm=False):
        """
        ref_feature_net gets entropy features from entropy

        x_in = [ temperature ]
        """

        layer_units = [1] + list(inner_layer_units) + [self.ref_feature_size]
        self.ref_feature_net_layer_units = layer_units
        self.ref_feature_net_batch_norm = batch_norm

        ref_feature_net_modules = []
        Layers = self.ref_feature_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_feature_net_modules.append(layer)
            if i < L-2:
                if self.ref_feature_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_feature_net_modules.append(layer)
                print("relu")
                ref_feature_net_modules.append(torch.nn.ReLU())
        self.ref_feature_net = nn.Sequential(*ref_feature_net_modules)
        return

    def build_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """

        layer_units = [self.entropy_input_size] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size+1]
        self.parameter_net_layer_units = layer_units
        self.parameter_net_batch_norm = batch_norm

        parameter_net_modules = []
        Layers = self.parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            parameter_net_modules.append(layer)
            if i < L-2:
                if self.parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    parameter_net_modules.append(layer)
                print("relu")
                parameter_net_modules.append(torch.nn.ReLU())
        self.parameter_net = nn.Sequential(*parameter_net_modules)

        return

    def build_ref_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        ref_parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """
        layer_units = [self.ref_input_size] + list(inner_layer_units)
        layer_units += [self.ref_feature_size+1]
        self.ref_parameter_net_layer_units = layer_units
        self.ref_parameter_net_batch_norm = batch_norm

        ref_parameter_net_modules = []
        Layers = self.ref_parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_parameter_net_modules.append(layer)
            if i < L-2:
                if self.ref_parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_parameter_net_modules.append(layer)
                print("relu")
                ref_parameter_net_modules.append(torch.nn.ReLU())
        self.ref_parameter_net = nn.Sequential(*ref_parameter_net_modules)

        return

    def forward_ref(self, x_in):
        print("xxx",x_in.shape)
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        dummy = self.ref_parameter_net(embedding)
        p_ref, temperature0 = torch.split(dummy, [self.ref_feature_size,1], 1)
        x_ref = self.ref_feature_net(temperature/temperature0)
        return torch.sum(p_ref*x_ref, dim=1)
    
    def forward(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        dummy = self.ref_parameter_net(embedding)
        p_ref, temperature0 = torch.split(dummy, [self.ref_feature_size,1], 1)
        x_ref = self.ref_feature_net(temperature/temperature0)

        # two independent parameter nets
        p_in = torch.cat([embedding, p_ref], 1)
        dummy = self.parameter_net(p_in)
        p_entropy, entropy0 = torch.split(dummy, [self.entropy_feature_size,1], 1)
        x_entropy = self.entropy_feature_net(entropy/entropy0)

        v_entropy = torch.sum(p_entropy*x_entropy, dim=1)
        v_ref = torch.sum(p_ref*x_ref, dim=1)
        return torch.unsqueeze(v_entropy + v_ref, 1)

class model0_ci_n2(torch.nn.Module):

    def __init__(self, embedding_size, entropy_feature_size, ref_feature_size):
        super(model0_ci_n2, self).__init__()

        self.embedding_size = embedding_size

        self.entropy_feature_size = entropy_feature_size
        self.entropy_input_size = embedding_size + ref_feature_size
        self.ref_feature_size = ref_feature_size
        self.ref_input_size = embedding_size
        return

    def build_entropy_feature_net(self, inner_layer_units, batch_norm=False):
        """
        entropy_feature_net gets entropy features from entropy

        x_in = [ entropy ]
        """

        layer_units = [1] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size]
        self.entropy_feature_net_layer_units = layer_units
        self.entropy_feature_net_batch_norm = batch_norm

        entropy_feature_net_modules = []
        Layers = self.entropy_feature_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            entropy_feature_net_modules.append(layer)
            if i < L-2:
                if self.entropy_feature_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    entropy_feature_net_modules.append(layer)
                print("relu")
                entropy_feature_net_modules.append(torch.nn.ReLU())
        self.entropy_feature_net = nn.Sequential(*entropy_feature_net_modules)
        # else:
        #     self.entropy_feature_net = self.get_polynomial_features
        return

    def build_ref_feature_net(self, inner_layer_units, batch_norm=False):
        """
        ref_feature_net gets entropy features from entropy

        x_in = [ temperature ]
        """

        layer_units = [1] + list(inner_layer_units) + [self.ref_feature_size]
        self.ref_feature_net_layer_units = layer_units
        self.ref_feature_net_batch_norm = batch_norm

        ref_feature_net_modules = []
        Layers = self.ref_feature_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_feature_net_modules.append(layer)
            if i < L-2:
                if self.ref_feature_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_feature_net_modules.append(layer)
                print("relu")
                ref_feature_net_modules.append(torch.nn.ReLU())
        self.ref_feature_net = nn.Sequential(*ref_feature_net_modules)
        return

    def build_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """

        layer_units = [self.entropy_input_size] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size+1]
        self.parameter_net_layer_units = layer_units
        self.parameter_net_batch_norm = batch_norm

        parameter_net_modules = []
        Layers = self.parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            parameter_net_modules.append(layer)
            if i < L-2:
                if self.parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    parameter_net_modules.append(layer)
                print("relu")
                parameter_net_modules.append(torch.nn.ReLU())
        self.parameter_net = nn.Sequential(*parameter_net_modules)

        return

    def build_ref_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        ref_parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """
        layer_units = [self.ref_input_size] + list(inner_layer_units)
        layer_units += [self.ref_feature_size+1]
        self.ref_parameter_net_layer_units = layer_units
        self.ref_parameter_net_batch_norm = batch_norm

        ref_parameter_net_modules = []
        Layers = self.ref_parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_parameter_net_modules.append(layer)
            if i < L-2:
                if self.ref_parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_parameter_net_modules.append(layer)
                print("relu")
                ref_parameter_net_modules.append(torch.nn.ReLU())
        self.ref_parameter_net = nn.Sequential(*ref_parameter_net_modules)

        return

    def forward_ref(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        dummy = self.ref_parameter_net(embedding)
        p_ref, temperature0 = torch.split(dummy, [self.ref_feature_size,1], 1)
        x_ref = self.ref_feature_net(temperature*temperature0)
        return torch.sum(p_ref*x_ref, dim=1)
    
    def forward(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        dummy = self.ref_parameter_net(embedding)
        p_ref, temperature0 = torch.split(dummy, [self.ref_feature_size,1], 1)
        x_ref = self.ref_feature_net(temperature*temperature0)

        # two independent parameter nets
        p_in = torch.cat([embedding, p_ref], 1)
        dummy = self.parameter_net(p_in)
        p_entropy, entropy0 = torch.split(dummy, [self.entropy_feature_size,1], 1)
        x_entropy = self.entropy_feature_net(entropy*entropy0)

        v_entropy = torch.sum(p_entropy*x_entropy, dim=1)
        v_ref = torch.sum(p_ref*x_ref, dim=1)
        return torch.unsqueeze(v_entropy + v_ref, 1)

class model0_ci_n2pe(torch.nn.Module):

    def __init__(self, embedding_size, entropy_feature_size, ref_feature_size):
        super(model0_ci_n2pe, self).__init__()

        self.embedding_size = embedding_size

        self.entropy_feature_size = entropy_feature_size
        self.entropy_input_size = embedding_size + ref_feature_size
        self.ref_feature_size = ref_feature_size
        self.ref_input_size = embedding_size

        start = 0
        step = 0.25        
        self.pows_entropy = torch.arange(start,start+self.entropy_feature_size*step, step)#.unsqueeze(0)        
        print(self.pows_entropy)
        return

    def build_ref_feature_net(self, inner_layer_units, batch_norm=False):
        """
        ref_feature_net gets entropy features from entropy

        x_in = [ temperature ]
        """

        layer_units = [1] + list(inner_layer_units) + [self.ref_feature_size]
        self.ref_feature_net_layer_units = layer_units
        self.ref_feature_net_batch_norm = batch_norm

        ref_feature_net_modules = []
        Layers = self.ref_feature_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_feature_net_modules.append(layer)
            if i < L-2:
                if self.ref_feature_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_feature_net_modules.append(layer)
                print("relu")
                ref_feature_net_modules.append(torch.nn.ReLU())
        self.ref_feature_net = nn.Sequential(*ref_feature_net_modules)
        return

    def build_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """

        layer_units = [self.entropy_input_size] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size+1]
        self.parameter_net_layer_units = layer_units
        self.parameter_net_batch_norm = batch_norm

        parameter_net_modules = []
        Layers = self.parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            parameter_net_modules.append(layer)
            if i < L-2:
                if self.parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    parameter_net_modules.append(layer)
                print("relu")
                parameter_net_modules.append(torch.nn.ReLU())
        self.parameter_net = nn.Sequential(*parameter_net_modules)

        return

    def build_ref_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        ref_parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """
        layer_units = [self.ref_input_size] + list(inner_layer_units)
        layer_units += [self.ref_feature_size+1]
        self.ref_parameter_net_layer_units = layer_units
        self.ref_parameter_net_batch_norm = batch_norm

        ref_parameter_net_modules = []
        Layers = self.ref_parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_parameter_net_modules.append(layer)
            if i < L-2:
                if self.ref_parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_parameter_net_modules.append(layer)
                print("relu")
                ref_parameter_net_modules.append(torch.nn.ReLU())
        self.ref_parameter_net = nn.Sequential(*ref_parameter_net_modules)

        return

    def forward_ref(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        dummy = self.ref_parameter_net(embedding)
        p_ref, temperature0 = torch.split(dummy, [self.ref_feature_size,1], 1)
        x_ref = self.ref_feature_net(temperature*temperature0)
        return torch.sum(p_ref*x_ref, dim=1)
    
    def forward(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        dummy = self.ref_parameter_net(embedding)
        p_ref, temperature0 = torch.split(dummy, [self.ref_feature_size,1], 1)
        x_ref = self.ref_feature_net(temperature*temperature0)

        # two independent parameter nets
        p_in = torch.cat([embedding, p_ref], 1)
        dummy = self.parameter_net(p_in)
        p_entropy, entropy0 = torch.split(dummy, [self.entropy_feature_size,1], 1)
        #x_entropy = self.entropy_feature_net(entropy*entropy0)
        #print(entropy.shape)
        #print(entropy)
        #print(entropy0.shape)
        #print(entropy0)
        x_entropy = entropy*entropy0
        x_entropy = torch.square(x_entropy)
        print(x_entropy.shape)
        x_entropy = torch.pow( x_entropy, self.pows_entropy )
        #print(x_entropy.shape)
        #print(x_entropy)
        #print()
        #print()
        #xxxxx
        v_entropy = torch.sum(p_entropy*x_entropy, dim=1)
        print(v_entropy.shape)
        v_ref = torch.sum(p_ref*x_ref, dim=1)
        print(v_ref.shape)
        print()
        return torch.unsqueeze(v_entropy + v_ref, 1)

class model0_ci_poly(torch.nn.Module):

    def __init__(self, embedding_size, entropy_feature_size, ref_feature_size):
        super(model0_ci_poly, self).__init__()

        self.embedding_size = embedding_size

        self.entropy_feature_size = entropy_feature_size
        self.entropy_input_size = embedding_size + ref_feature_size
        self.ref_feature_size = ref_feature_size
        self.ref_input_size = embedding_size

        start = 1
        step = 0.25        
        self.pows_entropy = torch.arange(start,start+self.entropy_feature_size*step, step)#.unsqueeze(0)
        
        return

    def get_polynomial_features(self, x_in, ppow, base=True):
        """Creates the polynomial features
        
        Args:
            x_in: A torch tensor for the data.
            ppow: pow steps.
            degree: self.entropy_feature_size.
        """
        degree = self.entropy_feature_size
        if base:
            ones_col = torch.ones( [x_in.shape[0], 1 ], dtype=torch.float32)
            x_out = torch.cat([ones_col, x_in], axis=1)
            degree -= 1
        else:
            x_out = x_in
        for i in range(1, degree):
            x_pow = x_in.pow(1+i*ppow)     
            x_out = torch.cat([x_out, x_pow], axis=1)
        return x_out

    def build_ref_feature_net(self, inner_layer_units, batch_norm=False):
        """
        ref_feature_net gets entropy features from entropy

        x_in = [ temperature ]
        """

        layer_units = [1] + list(inner_layer_units) + [self.ref_feature_size]
        self.ref_feature_net_layer_units = layer_units
        self.ref_feature_net_batch_norm = batch_norm

        ref_feature_net_modules = []
        Layers = self.ref_feature_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_feature_net_modules.append(layer)
            if i < L-2:
                if self.ref_feature_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_feature_net_modules.append(layer)
                print("relu")
                ref_feature_net_modules.append(torch.nn.ReLU())
        self.ref_feature_net = nn.Sequential(*ref_feature_net_modules)
        return

    def build_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """

        layer_units = [self.entropy_input_size] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size]
        self.parameter_net_layer_units = layer_units
        self.parameter_net_batch_norm = batch_norm

        parameter_net_modules = []
        Layers = self.parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            parameter_net_modules.append(layer)
            if i < L-2:
                if self.parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    parameter_net_modules.append(layer)
                print("relu")
                parameter_net_modules.append(torch.nn.ReLU())
        self.parameter_net = nn.Sequential(*parameter_net_modules)

        return

    def build_ref_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        ref_parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """
        layer_units = [self.ref_input_size] + list(inner_layer_units)
        layer_units += [self.ref_feature_size]
        self.ref_parameter_net_layer_units = layer_units
        self.ref_parameter_net_batch_norm = batch_norm

        ref_parameter_net_modules = []
        Layers = self.ref_parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_parameter_net_modules.append(layer)
            if i < L-2:
                if self.ref_parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_parameter_net_modules.append(layer)
                print("relu")
                ref_parameter_net_modules.append(torch.nn.ReLU())
        self.ref_parameter_net = nn.Sequential(*ref_parameter_net_modules)

        return

    def forward_ref(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        x_ref = self.ref_feature_net(temperature)
        p_ref = self.ref_parameter_net(embedding)
        return torch.sum(p_ref*x_ref, dim=1)
    
    def forward(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy0, embedding = dummy

        # two independent parameter nets
        p_ref = self.ref_parameter_net(embedding)
        p_in = torch.cat([embedding, p_ref], 1)
        #p_dummy = self.parameter_net(p_in)
        #p_entropy, eta = torch.split(p_dummy, [self.entropy_feature_size, 1], 1)
        p_entropy = self.parameter_net(p_in)

        x_ref = self.ref_feature_net(temperature)
        #x_entropy = self.entropy_feature_net(entropy)
        entropy = entropy0 #/ eta
        #x_entropy = self.get_polynomial_features(entropy,0.25)
        x_entropy = torch.pow( entropy, self.pows_entropy )

        v_entropy = torch.sum(p_entropy*x_entropy, dim=1)
        v_ref = torch.sum(p_ref*x_ref, dim=1)
        return torch.unsqueeze(v_entropy + v_ref, 1)


class model0_ci_poly_eta(torch.nn.Module):

    def __init__(self, embedding_size, entropy_feature_size, ref_feature_size):
        super(model0_ci_poly_eta, self).__init__()

        self.embedding_size = embedding_size

        self.entropy_feature_size = entropy_feature_size
        self.entropy_input_size = embedding_size + ref_feature_size
        self.ref_feature_size = ref_feature_size
        self.ref_input_size = embedding_size

        start = 1
        step = 0.25        
        self.pows_entropy = torch.arange(start,start+self.entropy_feature_size*step, step)#.unsqueeze(0)
        
        return

    def get_polynomial_features(self, x_in, ppow, base=True):
        """Creates the polynomial features
        
        Args:
            x_in: A torch tensor for the data.
            ppow: pow steps.
            degree: self.entropy_feature_size.
        """
        degree = self.entropy_feature_size
        if base:
            ones_col = torch.ones( [x_in.shape[0], 1 ], dtype=torch.float32)
            x_out = torch.cat([ones_col, x_in], axis=1)
            degree -= 1
        else:
            x_out = x_in
        for i in range(1, degree):
            x_pow = x_in.pow(1+i*ppow)     
            x_out = torch.cat([x_out, x_pow], axis=1)
        return x_out

    def build_ref_feature_net(self, inner_layer_units, batch_norm=False):
        """
        ref_feature_net gets entropy features from entropy

        x_in = [ temperature ]
        """

        layer_units = [1] + list(inner_layer_units) + [self.ref_feature_size]
        self.ref_feature_net_layer_units = layer_units
        self.ref_feature_net_batch_norm = batch_norm

        ref_feature_net_modules = []
        Layers = self.ref_feature_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_feature_net_modules.append(layer)
            if i < L-2:
                if self.ref_feature_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_feature_net_modules.append(layer)
                print("relu")
                ref_feature_net_modules.append(torch.nn.ReLU())
        self.ref_feature_net = nn.Sequential(*ref_feature_net_modules)
        return

    def build_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """

        layer_units = [self.entropy_input_size] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size+1]
        self.parameter_net_layer_units = layer_units
        self.parameter_net_batch_norm = batch_norm

        parameter_net_modules = []
        Layers = self.parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            parameter_net_modules.append(layer)
            if i < L-2:
                if self.parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    parameter_net_modules.append(layer)
                print("relu")
                parameter_net_modules.append(torch.nn.ReLU())
        self.parameter_net = nn.Sequential(*parameter_net_modules)

        return

    def build_ref_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        ref_parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """
        layer_units = [self.ref_input_size] + list(inner_layer_units)
        layer_units += [self.ref_feature_size]
        self.ref_parameter_net_layer_units = layer_units
        self.ref_parameter_net_batch_norm = batch_norm

        ref_parameter_net_modules = []
        Layers = self.ref_parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_parameter_net_modules.append(layer)
            if i < L-2:
                if self.ref_parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_parameter_net_modules.append(layer)
                print("relu")
                ref_parameter_net_modules.append(torch.nn.ReLU())
        self.ref_parameter_net = nn.Sequential(*ref_parameter_net_modules)

        return

    def forward_ref(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        x_ref = self.ref_feature_net(temperature)
        p_ref = self.ref_parameter_net(embedding)
        return torch.sum(p_ref*x_ref, dim=1)
    
    def forward(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy0, embedding = dummy

        # two independent parameter nets
        p_ref = self.ref_parameter_net(embedding)
        p_in = torch.cat([embedding, p_ref], 1)
        p_dummy = self.parameter_net(p_in)
        p_entropy, eta = torch.split(p_dummy, [self.entropy_feature_size, 1], 1)
        #p_entropy = self.parameter_net(p_in)

        x_ref = self.ref_feature_net(temperature)
        #x_entropy = self.entropy_feature_net(entropy)
        entropy = entropy0 * eta
        #x_entropy = self.get_polynomial_features(entropy,0.25)
        x_entropy = torch.pow( entropy, self.pows_entropy )

        v_entropy = torch.sum(p_entropy*x_entropy, dim=1)
        v_ref = torch.sum(p_ref*x_ref, dim=1)
        return torch.unsqueeze(v_entropy + v_ref, 1)

    
class model0_ci_poly2(torch.nn.Module):

    def __init__(self, embedding_size, entropy_feature_size, ref_feature_size):
        super(model0_ci_poly2, self).__init__()

        self.embedding_size = embedding_size

        self.entropy_feature_size = entropy_feature_size
        self.entropy_input_size = embedding_size + ref_feature_size
        self.ref_feature_size = ref_feature_size
        self.ref_input_size = embedding_size

        start = 0
        step = 0.5
        self.pows_ref = torch.arange(start, start+self.ref_feature_size*step, step)#.unsqueeze(0)
        start = 1
        step = 0.25        
        self.pows_entropy = torch.arange(start,start+self.entropy_feature_size*step, step)#.unsqueeze(0)
        
        return


    def build_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """

        layer_units = [self.entropy_input_size] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size]
        self.parameter_net_layer_units = layer_units
        self.parameter_net_batch_norm = batch_norm

        parameter_net_modules = []
        Layers = self.parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            parameter_net_modules.append(layer)
            if i < L-2:
                if self.parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    parameter_net_modules.append(layer)
                print("relu")
                parameter_net_modules.append(torch.nn.ReLU())
        self.parameter_net = nn.Sequential(*parameter_net_modules)

        return

    def build_ref_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        ref_parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """
        layer_units = [self.ref_input_size] + list(inner_layer_units)
        layer_units += [self.ref_feature_size]
        self.ref_parameter_net_layer_units = layer_units
        self.ref_parameter_net_batch_norm = batch_norm

        ref_parameter_net_modules = []
        Layers = self.ref_parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_parameter_net_modules.append(layer)
            if i < L-2:
                if self.ref_parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_parameter_net_modules.append(layer)
                print("relu")
                ref_parameter_net_modules.append(torch.nn.ReLU())
        self.ref_parameter_net = nn.Sequential(*ref_parameter_net_modules)

        return

    def forward_ref(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        #x_ref = self.ref_feature_net(temperature)
        #x_ref = self.get_polynomial_features(temperature,0.25,self.ref_feature_size)
        x_ref = torch.pow( temperature, self.pows_ref )
        p_ref = self.ref_parameter_net(embedding)
        return torch.sum(p_ref*x_ref, dim=1)
    
    def forward(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy0, embedding = dummy

        # two independent parameter nets
        p_ref = self.ref_parameter_net(embedding)
        p_in = torch.cat([embedding, p_ref], 1)
        p_entropy = self.parameter_net(p_in)

        #x_ref = self.ref_feature_net(temperature)
        #x_entropy = self.entropy_feature_net(entropy)
        entropy = entropy0 #/ eta
        #x_entropy = self.get_polynomial_features(entropy,0.25,self.entropy_feature_size,base=False)
        #x_ref = self.get_polynomial_features(temperature,0.25,self.ref_feature_size)
        x_entropy = torch.pow( entropy, self.pows_entropy )
        x_ref = torch.pow( temperature, self.pows_ref )
        #print( "x", x_entropy.shape, "p", p_entropy.shape )
        
        v_entropy = torch.sum(p_entropy*x_entropy, dim=1)
        v_ref = torch.sum(p_ref*x_ref, dim=1)
        return torch.unsqueeze(v_entropy + v_ref, 1)



class model0_ci_poly2_eta(torch.nn.Module):

    def __init__(self, embedding_size, entropy_feature_size, ref_feature_size):
        super(model0_ci_poly2_eta, self).__init__()

        self.embedding_size = embedding_size

        self.entropy_feature_size = entropy_feature_size
        self.entropy_input_size = embedding_size + ref_feature_size
        self.ref_feature_size = ref_feature_size
        self.ref_input_size = embedding_size

        start = 0
        step = 0.5
        self.pows_ref = torch.arange(start, start+self.ref_feature_size*step, step)#.unsqueeze(0)
        start = 0.25
        step = 0.25        
        self.pows_entropy = torch.arange(start,start+self.entropy_feature_size*step, step)#.unsqueeze(0)
        
        return


    def build_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """

        layer_units = [self.entropy_input_size] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size+1]
        self.parameter_net_layer_units = layer_units
        self.parameter_net_batch_norm = batch_norm

        parameter_net_modules = []
        Layers = self.parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            parameter_net_modules.append(layer)
            if i < L-2:
                if self.parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    parameter_net_modules.append(layer)
                print("relu")
                parameter_net_modules.append(torch.nn.ReLU())
        self.parameter_net = nn.Sequential(*parameter_net_modules)

        return

    def build_ref_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        ref_parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """
        layer_units = [self.ref_input_size] + list(inner_layer_units)
        layer_units += [self.ref_feature_size]
        self.ref_parameter_net_layer_units = layer_units
        self.ref_parameter_net_batch_norm = batch_norm

        ref_parameter_net_modules = []
        Layers = self.ref_parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_parameter_net_modules.append(layer)
            if i < L-2:
                if self.ref_parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_parameter_net_modules.append(layer)
                print("relu")
                ref_parameter_net_modules.append(torch.nn.ReLU())
        self.ref_parameter_net = nn.Sequential(*ref_parameter_net_modules)

        return

    def forward_ref(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        #x_ref = self.ref_feature_net(temperature)
        #x_ref = self.get_polynomial_features(temperature,0.25,self.ref_feature_size)
        x_ref = torch.pow( temperature, self.pows_ref )
        p_ref = self.ref_parameter_net(embedding)
        return torch.sum(p_ref*x_ref, dim=1)
    
    def forward(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy0, embedding = dummy

        # two independent parameter nets
        p_ref = self.ref_parameter_net(embedding)
        p_in = torch.cat([embedding, p_ref], 1)
        p_dummy = self.parameter_net(p_in)
        p_entropy, eta = torch.split(p_dummy, [self.entropy_feature_size, 1], 1)
        #p_entropy = self.parameter_net(p_in)

        #x_ref = self.ref_feature_net(temperature)
        #x_entropy = self.entropy_feature_net(entropy)
        entropy = entropy0 * eta
        #x_entropy = self.get_polynomial_features(entropy,0.25,self.entropy_feature_size,base=False)
        #x_ref = self.get_polynomial_features(temperature,0.25,self.ref_feature_size)
        x_entropy = torch.pow( entropy, self.pows_entropy )
        x_ref = torch.pow( temperature, self.pows_ref )
        #print( "x", x_entropy.shape, "p", p_entropy.shape )
        
        v_entropy = torch.sum(p_entropy*x_entropy, dim=1)
        v_ref = torch.sum(p_ref*x_ref, dim=1)
        return torch.unsqueeze(v_entropy + v_ref, 1)


class model0_ci_p2n2(torch.nn.Module):

    def __init__(self, embedding_size, entropy_feature_size, ref_feature_size):
        super(model0_ci_p2n2, self).__init__()

        self.embedding_size = embedding_size

        self.entropy_feature_size = entropy_feature_size
        self.entropy_input_size = embedding_size + ref_feature_size
        self.ref_feature_size = ref_feature_size
        self.ref_input_size = embedding_size

        start = 1
        step = 0.5
        self.pows_ref = torch.arange(start, start+self.ref_feature_size*step, step)#.unsqueeze(0)
        start = 1 #0.25
        step = 0.25        
        self.pows_entropy = torch.arange(start,start+self.entropy_feature_size*step, step)#.unsqueeze(0)
        
        return


    def build_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """

        layer_units = [self.entropy_input_size] + list(inner_layer_units)
        layer_units += [self.entropy_feature_size+1]
        self.parameter_net_layer_units = layer_units
        self.parameter_net_batch_norm = batch_norm

        parameter_net_modules = []
        Layers = self.parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            parameter_net_modules.append(layer)
            if i < L-2:
                if self.parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    parameter_net_modules.append(layer)
                print("relu")
                parameter_net_modules.append(torch.nn.ReLU())
        self.parameter_net = nn.Sequential(*parameter_net_modules)

        return

    def build_ref_parameter_net(self, inner_layer_units, batch_norm=False):
        """
        ref_parameter_net predicts parameters from moldecule embedding

        the parameters activate the entropy features

        x_in = [ *depends on build ]
        """
        layer_units = [self.ref_input_size] + list(inner_layer_units)
        layer_units += [self.ref_feature_size+1]
        self.ref_parameter_net_layer_units = layer_units
        self.ref_parameter_net_batch_norm = batch_norm

        ref_parameter_net_modules = []
        Layers = self.ref_parameter_net_layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            ref_parameter_net_modules.append(layer)
            if i < L-2:
                if self.ref_parameter_net_batch_norm:
                    print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    ref_parameter_net_modules.append(layer)
                print("relu")
                ref_parameter_net_modules.append(torch.nn.ReLU())
        self.ref_parameter_net = nn.Sequential(*ref_parameter_net_modules)

        return

    def forward_ref(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        dummy = self.ref_parameter_net(embedding)
        p_ref, temperature0 = torch.split(dummy, [self.ref_feature_size,1], 1)
        temperature = temperature*temperature0
        x_ref = torch.pow( temperature, self.pows_ref )
        return torch.sum(p_ref*x_ref, dim=1)
    
    def forward(self, x_in):
        dummy = torch.split(x_in, [1, 1, self.ref_input_size], 1)
        temperature, entropy, embedding = dummy
        dummy = self.ref_parameter_net(embedding)
        p_ref, temperature0 = torch.split(dummy, [self.ref_feature_size,1], 1)
        temperature = temperature*temperature0
        x_ref = torch.pow( temperature, self.pows_ref )
        
        p_in = torch.cat([embedding, p_ref], 1)
        p_dummy = self.parameter_net(p_in)
        p_entropy, entropy0 = torch.split(p_dummy, [self.entropy_feature_size, 1], 1)
        #print(entropy0.shape, eta.shape)
        entropy = entropy* entropy0
        #print(entropy.shape)
        x_entropy = torch.pow( entropy, self.pows_entropy )
        
        v_entropy = torch.sum(p_entropy*x_entropy, dim=1)
        v_ref = torch.sum(p_ref*x_ref, dim=1)
        return torch.unsqueeze(v_entropy + v_ref, 1)
