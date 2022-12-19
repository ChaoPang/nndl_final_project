import random

import torch.nn.functional
from ensemble_trainer import *
from models.finetune_pretrained import PretrainedModel

MODEL_NAME = 'best_model.pt'

sys.setrecursionlimit(10000)


def map_idx_to_superclass(
        predictions: pd.Series
):
    return predictions.apply(IDX_TO_SUPERCLASS_DICT.get)


def map_idx_to_subclass(
        predictions: pd.Series
):
    return predictions.apply(IDX_TO_SUBCLASS_MAPPING.get)


def create_arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch NNDL image classification challenge')
    parser.add_argument('--is_superclass', action='store_true',
                        help='Super class or sub class predictions')
    parser.add_argument('--num_of_classifiers', type=int, default=5, help='num_of_classifiers')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='The number of superclass categories')
    parser.add_argument('--num_subclasses', type=int, default=90,
                        help='The number of subclasse categories')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument(
        '--pretrained_model',
        action='store',
        choices=[e.value for e in PretrainedModel],
        default=PretrainedModel.FinetuneEfficientNetV2FeatureExtractor.value
    )
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='Dropout rate')
    parser.add_argument('--train_percentage', default=0.8, type=float,
                        help='training_percentage')
    parser.add_argument('--freeze_weight', action='store_true',
                        help='Whether or not we freeze the weights of the pretrained model')
    parser.add_argument('--deep_feature', action='store_true',
                        help='Whether or not extract the deep feature')
    parser.add_argument('--early_stopping_patience', default=10, type=int,
                        help='Early stopping patience')
    parser.add_argument('--img_size', default=8, type=int, help='Image Size')
    parser.add_argument('--training_data_path', required=True,
                        help='input_folder containing the images')
    parser.add_argument('--training_label_path', required=True, help='the path to training label')
    parser.add_argument('--test_data_path', required=True,
                        help='input_folder containing the images')
    parser.add_argument('--checkpoint_path', required=True, help='checkpoint_path for the model')
    parser.add_argument('--external_validation', action='store_true',
                        help='Using CIFAR data to validate the model')
    parser.add_argument('--use_cifar_for_test', action='store_true',
                        help='Using CIFAR data to test the model')
    parser.add_argument('--val_data_path', required='--external_validation' in sys.argv,
                        help='input_folder containing the CIFAR images')
    return parser


def create_training_datasets(
        args
):
    train_set = ProjectDataSet(
        image_folder_path=args.training_data_path,
        data_label_path=args.training_label_path,
        is_training=True,
        is_superclass=args.is_superclass,
        img_size=args.img_size,
        multitask=True
    )

    # If the up sampler is enabled, we use the default 32 by 32 image for validation
    # Use the CIFAR data as the external validation set
    if args.external_validation:
        val_set = CifarValidationDataset(
            cifar_data_folder=args.val_data_path,
            download=True,
            img_size=args.img_size,
            multitask=True
        )
        # Randomly slice out data for training to inject more noise into the ensemble method
        train_size = int(len(train_set) * args.train_percentage)
        train_set, _ = torch.utils.data.random_split(
            train_set, [train_size, len(train_set) - train_size]
        )
    else:
        train_total = len(train_set)
        train_size = int(train_total * args.train_percentage)
        val_size = train_total - train_size
        train_set, val_set = torch.utils.data.random_split(
            train_set, [train_size, val_size]
        )

    print(f'train_set size: {len(train_set)}')
    print(f'val_set size: {len(val_set)}')

    return train_set, val_set


# Training
def train(
        net,
        train_loader,
        superclass_criterion,
        subclass_criterion,
        optimizer
):
    net.train()

    train_loss = 0
    superclass_correct = 0
    subclass_correct = 0
    total = 0

    for batch_idx, (inputs, superclass_targets, subclass_targets) in enumerate(train_loader):
        inputs = inputs.to(get_device())
        superclass_targets = superclass_targets.to(get_device())
        subclass_targets = subclass_targets.to(get_device())

        optimizer.zero_grad()

        superclass_outputs, subclass_outputs = net(inputs)
        superclass_loss = superclass_criterion(superclass_outputs, superclass_targets)
        subclass_loss = subclass_criterion(subclass_outputs, subclass_targets)

        loss = superclass_loss + subclass_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += subclass_targets.size(0)

        _, superclass_predicted = superclass_outputs.max(1)
        superclass_correct += superclass_predicted.eq(superclass_targets).sum().item()

        _, subclass_predicted = subclass_outputs.max(1)
        subclass_correct += subclass_predicted.eq(subclass_targets).sum().item()

        progress_bar(
            batch_idx, len(train_loader),
            'Superclass Train Loss: %.3f | Subclass Train Loss: %.3f | '
            'Superclass Train Acc: %.3f%% (%d/%d) | Subclass Train Acc: %.3f%% (%d/%d)'
            % (
                superclass_loss,
                subclass_loss,
                100. * superclass_correct / total, superclass_correct, total,
                100. * subclass_correct / total, subclass_correct, total,
            )
        )

    subclass_acc = 100. * subclass_correct / total
    train_average_loss = train_loss / total
    return train_average_loss, subclass_acc


def validate(
        net,
        val_loader,
        superclass_criterion,
        subclass_criterion,
        external_validation
):
    net.eval()
    val_loss = 0
    subclass_correct = 0
    superclass_correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, superclass_targets, subclass_targets) in enumerate(val_loader):
            inputs = inputs.to(get_device())
            superclass_targets = superclass_targets.to(get_device())
            subclass_targets = subclass_targets.to(get_device())

            superclass_outputs, subclass_outputs = net(inputs)
            superclass_loss = superclass_criterion(superclass_outputs, superclass_targets)

            loss = superclass_loss
            total += superclass_outputs.size(0)

            _, superclass_predictions = superclass_outputs.max(1)
            superclass_correct += superclass_predictions.eq(superclass_targets).sum().item()

            # In case of external validations, we don't have labels for the subclass prediction
            # Let's only focus on the super class prediction
            if not external_validation:
                subclass_loss = subclass_criterion(subclass_outputs, subclass_targets)
                loss = loss + subclass_loss
                _, subclass_predictions = subclass_outputs.max(1)
                subclass_correct += subclass_predictions.eq(subclass_targets).sum().item()
            else:
                subclass_loss = 0

            val_loss += loss.item()

            progress_bar(
                batch_idx, len(val_loader),
                'Superclass Val Loss: %.3f | Subclass Val Loss: %.3f | '
                'Superclass val Acc: %.3f%% (%d/%d) | Subclass Val Acc: %.3f%% (%d/%d)'
                % (
                    superclass_loss,
                    subclass_loss,
                    100. * superclass_correct / total, superclass_correct, total,
                    100. * subclass_correct / total, subclass_correct, total,
                )
            )

    subclass_acc = 100. * subclass_correct / total
    val_average_loss = val_loss / total
    return val_average_loss, subclass_acc


def predict(
        ensemble_models,
        args
):
    if args.use_cifar_for_test:
        test_set = CifarValidationDataset(
            cifar_data_folder=args.val_data_path,
            download=True,
            img_size=args.img_size,
            multitask=True
        )
    else:
        test_set = ProjectDataSet(
            image_folder_path=args.test_data_path,
            is_training=False,
            is_superclass=args.is_superclass,
            img_size=args.img_size,
            multitask=True
        )

    data_loader = DataLoader(
        test_set, batch_size=args.batch_size, num_workers=4
    )

    superclass_predictions = []
    subclass_predictions = []
    superclass_labels = []
    subclass_labels = []
    subclass_correct = 0
    superclass_correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, superclass_targets, subclass_targets) in enumerate(data_loader):

            outputs = []
            for net in ensemble_models:
                net.to(get_device())
                net.eval()
                outputs.append(net(inputs.to(get_device())))
                net.to('cpu')

            superclass_outputs, subclass_outputs = zip(*outputs)

            average_superclass_output = torch.stack(superclass_outputs, dim=1).mean(dim=1)
            average_subclass_output = torch.stack(subclass_outputs, dim=1).mean(dim=1)

            superclass_prediction_batch = torch.argmax(average_superclass_output, dim=-1)
            subclass_prediction_batch = torch.argmax(average_subclass_output, dim=-1)

            superclass_predictions.append(superclass_prediction_batch.detach().cpu().numpy())
            subclass_predictions.append(subclass_prediction_batch.detach().cpu().numpy())

            superclass_labels.append(superclass_targets.detach().cpu().numpy())
            subclass_labels.append(subclass_targets.detach().cpu().numpy())

            total += subclass_targets.size(0)
            superclass_correct += superclass_prediction_batch.detach().cpu().eq(
                superclass_targets).sum().item()
            subclass_correct += subclass_prediction_batch.detach().cpu().eq(
                subclass_targets).sum().item()

            progress_bar(batch_idx, len(data_loader),
                         'Superclass Acc: %.3f%% (%d/%d) | Subclass Acc: %.3f%% (%d/%d)'
                         % (100. * superclass_correct / total, superclass_correct, total,
                            100. * subclass_correct / total, subclass_correct, total))

        predictions_pd = pd.DataFrame(
            np.hstack(superclass_predictions),
            columns=['superclass_prediction']
        )
        predictions_pd['subclass_prediction'] = np.hstack(subclass_predictions)
        predictions_pd['prediction_class'] = map_idx_to_superclass(
            predictions_pd.superclass_prediction)
        predictions_pd['prediction_subclass'] = map_idx_to_subclass(
            predictions_pd.subclass_prediction)

    return predictions_pd


def train_model(
        ensemble_models,
        args
):
    histories = []
    # Train each net individually
    for net in ensemble_models:
        train_set, val_set = create_training_datasets(args)

        train_dataloader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
        )

        val_dataloader = DataLoader(
            val_set, batch_size=args.batch_size, shuffle=True, num_workers=4
        )

        net = net.to(get_device())

        weight_decay = random.uniform(0.5e-4, 1.5e-4)
        epsilon = random.uniform(0.01, 0.1)

        gamma = random.uniform(0.85, 0.95)

        superclass_criterion = nn.CrossEntropyLoss()
        subclass_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            net.parameters(), lr=args.lr, weight_decay=weight_decay, eps=epsilon
        )

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        print(f'Training {net.name}; weight_decay={weight_decay}; epsilon={epsilon}; gamma={gamma}')

        history = training_loop(
            net,
            train_dataloader,
            val_dataloader,
            superclass_criterion,
            subclass_criterion,
            optimizer,
            scheduler,
            args.epochs,
            args.early_stopping_patience,
            args.checkpoint_path,
            args.external_validation
        )

        # Load the best model according to the val loss
        net = torch.load(
            os.path.join(args.checkpoint_path, net.name, MODEL_NAME),
            map_location=get_device()
        )

        history['name'] = net.name

        histories.append(history)

    return histories


def main(args):
    random_dropout_rates = [
        random.uniform(0.1, 0.3) for _ in range(args.num_of_classifiers)
    ]
    ensemble_models = [
        create_multitask_trainer(
            num_classes=args.num_classes,
            num_subclasses=args.num_subclasses,
            pretrained_model=PretrainedModel(args.pretrained_model),
            dropout_rate=random_dropout_rates[i],
            deep_feature=args.deep_feature,
            freeze_weight=args.freeze_weight,
            name=f'{args.pretrained_model}_{i}'
        ) for i in range(args.num_of_classifiers)
    ]

    histories = train_model(
        ensemble_models,
        args
    )

    predictions_pd = predict(
        ensemble_models,
        args
    )
    predictions_pd.to_csv(
        os.path.join(args.checkpoint_path, 'predictions.csv'),
        index=False
    )
    for history in histories:
        plot_training_loss(history, os.path.join(args.checkpoint_path, history['name']))


def training_loop(
        net,
        train_dataloader,
        val_dataloader,
        superclass_criterion,
        subclass_criterion,
        optimizer,
        scheduler,
        epochs,
        early_stopping_patience,
        checkpoint_path,
        external_validation
):
    early_stopping_counter = 0
    best_val_acc = 0

    history = {}

    for epoch in range(0, epochs):

        train_loss, train_acc = train(
            net,
            train_dataloader,
            superclass_criterion,
            subclass_criterion,
            optimizer
        )

        val_loss, val_acc = validate(
            net,
            val_dataloader,
            superclass_criterion,
            subclass_criterion,
            external_validation
        )
        scheduler.step()

        update_metrics(
            history,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc
        )

        if val_acc > best_val_acc:
            checkpoint(net, history, os.path.join(checkpoint_path, net.name), MODEL_NAME)
            best_val_acc = val_acc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Stop the training if the val does not improve
        if early_stopping_counter > early_stopping_patience:
            print("Validation loss has not improved in {} epochs, stopping early".format(
                early_stopping_patience))
            print("Obtained lowest validation loss of: {}".format(best_val_acc))
            break

    return history


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
