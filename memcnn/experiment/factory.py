import json
import copy


def get_attr_from_module(pclass):
    pclass = pclass.rsplit(".", 1)
    mod = __import__(pclass[0], fromlist=[str(pclass[1])])
    return getattr(mod, pclass[1])


def load_experiment_config(experiments_file, experiment_tags):
    with open(experiments_file, 'r') as f:
        data = json.load(f)
    d = {}
    for tag in experiment_tags:
        _inject_items(build_dict(data, tag), d)

    return d


def _inject_items(tempdict, d):
    """inject tempdict into d"""
    for k, v in tempdict.items():
        if isinstance(v, dict):
            if k not in d:
                d[k] = {}
            d[k] = _inject_items(v, d[k])
        else:
            d[k] = v
    return d


def build_dict(experiments_dict, experiment_name, classhist=None):
    tempdict = experiments_dict[experiment_name]
    if classhist is None:
        classhist = []
    classhist.append(experiment_name)
    if not ('base' in tempdict) or (tempdict['base'] is None):
        return copy.deepcopy(tempdict)
    elif tempdict['base'] in classhist:
        raise RuntimeError('Circular dependency found...')
    else:
        d = build_dict(experiments_dict, tempdict['base'], classhist)
        return _inject_items(tempdict, d)


def experiment_config_parser(d, data_dir, workers=None):
    trainer = get_attr_from_module(d['trainer'])

    model = get_attr_from_module(d['model'])
    model_params = copy.deepcopy(d['model_params'])
    if 'block' in model_params:
        model_params['block'] = get_attr_from_module(model_params['block'])
    model = model(**model_params)

    optimizer = get_attr_from_module(d['optimizer'])
    optimizer = optimizer(model.parameters(), **d['optimizer_params'])

    dl_params = copy.deepcopy(d['data_loader_params'])
    dl_params['dataset'] = get_attr_from_module(dl_params['dataset'])
    dl_params['data_dir'] = data_dir
    dl_params['workers'] = dl_params['workers'] if workers is None else workers

    train_loader, val_loader = get_attr_from_module(d['data_loader'])(**dl_params)

    trainer_params = {}
    if 'trainer_params' in d:
        trainer_params = copy.deepcopy(d['trainer_params'])
        if 'loss' in trainer_params:
            trainer_params['loss'] = get_attr_from_module(trainer_params['loss'])()

    trainer_params = dict(
        train_loader=train_loader,
        test_loader=val_loader,
        **trainer_params
    )

    return model, optimizer, trainer, trainer_params
