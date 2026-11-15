import yaml
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def load_config(args):
    """ Read the yaml config, apply command Line overrides, and prepare the save folder.

    Args:
        args: Parsed command line arguments.
    
    Returns:
        config: Dictionary containing the configuration for training.
    """
    with open(args.config,encoding='utf-8') as f:
        config=yaml.safe_load(f)

    # override config with command line arguments
    if args.arch is not None:
        config['arch']['model_name']=args.arch
    if args.lr is not None:
        config['optimizer']['args']['lr']=args.lr
    if args.epochs is not None:
        config['trainer']['epochs']=args.epochs

    # prepare the save folder and update paths in the config
    for key in ('csv_file','data_dir'):
        config['data_loader'][key]=str((ROOT/config['data_loader'][key]).resolve())
    save_dir=(ROOT/config['trainer']['save_dir']/config['arch']['model_name']).resolve()
    save_dir.mkdir(parents=True,exist_ok=True)
    config['trainer']['save_dir']=str(save_dir)

    # save a copy of the config for reproducibility
    with open(save_dir/'config.yaml','w',encoding='utf-8') as f:
        yaml.safe_dump(config,f,sort_keys=False,allow_unicode=True)
    return config

def init_obj(config,name,module,*args,**kwargs):
    """ Create an object from config[name], which has a 'type' and its 'args'.
    
    Args:
        config: Dictionary containing the configuration.
        name: Key in the config dictionary to look for the object type and args.
        module: Module that contains the class, such as torch.optim.
        *args: Positional arguments passed before the config args.
        **kwargs: Additional keyword arguments that are not in the config file.

    Returns:
        getattr(module,config[name]['type'])(*args,**config[name]['args'],**kwargs)
    """
    module_name=config[name]['type']
    module_args=dict(config[name]['args'])      # shallow copy to avoid modifying the original config
    module_args.update(kwargs)
    return getattr(module,module_name)(*args,**module_args)