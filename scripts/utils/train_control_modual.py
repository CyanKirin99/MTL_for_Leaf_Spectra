

class UnFreezer:
    def __init__(self, model, module_names, unfreeze_interval=20, always_train_modules=None):
        self.model = model
        self.module_names = module_names
        self.unfreeze_interval = unfreeze_interval
        self.current_interval = 0
        self.always_train_modules = always_train_modules if always_train_modules else []

        for name, param in self.model.named_parameters():
            if any(name.startswith(module) for module in self.always_train_modules):
                param.requires_grad = True
                print(f'Unfreeze Layer:\t{name}')

    def step(self, have_trained_epoch):
        if have_trained_epoch % self.unfreeze_interval == 0 and self.current_interval < len(self.module_names):
            module_name = self.module_names[self.current_interval]
            module_name_parts = module_name.split('.')
            for name, param in self.model.named_parameters():
                if len(module_name_parts) == 1:
                    if name.startswith(module_name):
                        param.requires_grad = True
                        print(f'Unfreeze Layer:\t{name}')
                else:
                    if name.startswith(module_name_parts[0]):
                        appendix_name = name.split(f'{module_name_parts[0]}.')[1]
                        if appendix_name.split('.')[0] == module_name_parts[1] or appendix_name.split('.')[1] == module_name_parts[1]:
                            param.requires_grad = True
                            print(f'Unfreeze Layer:\t{name}')
            self.current_interval += 1
