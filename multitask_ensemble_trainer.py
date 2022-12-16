import torch.nn.functional
from ensemble_trainer import *

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
    parser.add_argument('--num_classes', type=int, default=3, help='num_classes')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='Dropout rate')
    parser.add_argument('--freeze_weight', action='store_true',
                        help='Whether or not we freeze the weights of the pretrained model')
    parser.add_argument('--deep_feature', action='store_true',
                        help='Whether or not extract the deep feature')
    parser.add_argument('--multitask', action='store_true')
    parser.add_argument('--early_stopping_patience', default=10, type=int,
                        help='Early stopping patience')
    parser.add_argument('--img_size', default=8, type=int, help='Image Size')
    parser.add_argument('--training_data_path', required=True,
                        help='input_folder containing the images')
    parser.add_argument('--training_label_path', required=True, help='the path to training label')
    parser.add_argument('--test_data_path', required=True,
                        help='input_folder containing the images')
    parser.add_argument('--val_data_path', required=True,
                        help='input_folder containing the CIFAR images')
    parser.add_argument('--checkpoint_path', required=True, help='checkpoint_path for the model')
    parser.add_argument('--external_validation', action='store_true',
                        help='Using CIFAR data to test the model')
    parser.add_argument('--test_label', action='store_true',
                        help='Indicate whether the test label is available')

    return parser


# Training
def train(
        net,
        train_loader,
        loss_functions,
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
        superclass_loss = loss_functions[0](superclass_outputs, superclass_targets)
        subclass_loss = loss_functions[1](subclass_outputs, subclass_targets)

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
        loss_functions
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

            superclass_loss = loss_functions[0](superclass_outputs, superclass_targets)
            subclass_loss = loss_functions[1](subclass_outputs, subclass_targets)

            loss = superclass_loss + subclass_loss
            val_loss += loss.item()

            total += subclass_outputs.size(0)
            _, subclass_predictions = subclass_outputs.max(1)
            subclass_correct += subclass_predictions.eq(subclass_targets).sum().item()
            _, superclass_predictions = superclass_outputs.max(1)
            superclass_correct += superclass_predictions.eq(superclass_targets).sum().item()

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
    test_set = ProjectDataSet(
        image_folder_path=args.test_data_path,
        is_training=False,
        is_superclass=args.is_superclass,
        img_size=args.img_size,
        multitask=args.multitask
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

        if args.is_superclass:
            # Empirical evidence
            weight_decay = random.uniform(1e-3, 1e-4)
            epsilon = random.uniform(0.01, 0.1)
        else:
            weight_decay = random.uniform(0.5e-3, 1.5e-3)
            epsilon = random.uniform(0.05, 0.15)

        gamma = random.uniform(0.85, 0.95)

        loss_functions = [
            nn.CrossEntropyLoss(),
            nn.CrossEntropyLoss()
        ]

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
            loss_functions,
            optimizer,
            scheduler,
            args.epochs,
            args.early_stopping_patience,
            args.checkpoint_path
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
    ensemble_models = [
        FinetuneEfficientNetV2MultiTask(
            num_classes=3,
            num_sub_classes=90,
            deep_feature=args.deep_feature,
            freeze_weight=args.freeze_weight,
            name=f'FinetuneEfficientNetV2_{i}'
        ) for i in range(1)
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
        loss_functions,
        optimizer,
        scheduler,
        epochs,
        early_stopping_patience,
        checkpoint_path
):
    early_stopping_counter = 0
    best_val_loss = 1e6

    history = {}

    for epoch in range(0, epochs):

        train_loss, train_acc = train(
            net,
            train_dataloader,
            loss_functions,
            optimizer
        )

        val_loss, val_acc = validate(
            net,
            val_dataloader,
            loss_functions
        )
        scheduler.step()

        update_metrics(
            history,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc
        )

        if val_loss < best_val_loss:
            checkpoint(net, history, os.path.join(checkpoint_path, net.name), MODEL_NAME)
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Stop the training if the val does not improve
        if early_stopping_counter > early_stopping_patience:
            print("Validation loss has not improved in {} epochs, stopping early".format(
                early_stopping_patience))
            print("Obtained lowest validation loss of: {}".format(best_val_loss))
            break

    return history


if __name__ == "__main__":
    main(create_arg_parser().parse_args())
