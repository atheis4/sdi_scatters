import argparse
from datetime import datetime
import os
import time
import warnings

from cod_epi_data import (Cod,
                          Epi,
                          Sev,
                          CodLabels,
                          EpiLabels,
                          SevLabels,
                          Metadata,
                          SdiScatterPlot)


def pretty_now():
    return datetime.now().strftime("[%m/%d/%Y %H:%M:%S]")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gbd_team', type=str, required=True,
                        help="'cod', 'epi', 'exp', or 'sev' (required)")
    parser.add_argument('--model_version_id', type=int, default=None,
                        help=("Specific model version id to plot --\n"
                              "if not specified, must provide gbd_id and  "
                              "will default to best."))
    parser.add_argument('--gbd_id', type=int, default=None,
                        help=("'cause_id' or 'modelable_entity_id' as int --\n"
                              "defaults to best model"))
    parser.add_argument('--measure_id', type=int, default=None,
                        help=("Specific measure_id to plot "
                              "(epi only - required)"))
    parser.add_argument('--sex_id', type=int, default=None,
                        help=("Specifies which sex to plot --\n"
                              "(cod - required if no model_version provided)"))
    parser.add_argument('--age_group_id', type=int, nargs='+', default=None,
                        help=("Specifies the age group id(s) to plot"))
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Location in filesystem to save plot (required)")

    args = parser.parse_args()
    gbd_team = args.gbd_team
    output_dir = args.output_dir
    model_version_id = args.model_version_id
    gbd_id = args.gbd_id
    sex_id = args.sex_id
    age_group_id = args.age_group_id
    measure_id = args.measure_id

    return (gbd_team, output_dir, model_version_id, gbd_id, sex_id,
            age_group_id, measure_id)


def plot_and_save(plot, outdir):
    print("{} Plot".format(pretty_now()))
    plot.graph()
    print("{} Saving plot".format(pretty_now()))
    plot.save(outdir)
    print("{} Save successful".format(pretty_now()))


def create_scatter(gbd_team,
                   output_dir,
                   model_version_id=None,
                   gbd_id=None,
                   sex_id=None,
                   age_group_id=None,
                   measure_id=None):
    """Generates a country level scatter plot of socio-demographic index to
    model results for GBD 2017 Review Week."""

    outdir = output_dir
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    print("{} Validating arguments".format(pretty_now()))
    if gbd_team not in ['cod', 'epi', 'exp', 'sev']:
        raise ValueError("{} gbd_team parameter must be 'cod', 'epi', 'exp',"
                         " or 'sev'.".format(pretty_now))

    if not model_version_id:
        if gbd_team == 'cod' and not (gbd_id and sex_id):
            raise ValueError("{} If no model_version_id, must provide "
                             "a cause_id as gbd_id and a sex_id."
                             .format(pretty_now()))
        elif gbd_team in ['epi', 'exp'] and not gbd_id:
            raise ValueError("{} If no model_version_id, must provide a "
                             "modelable_entity_id as gbd_id."
                             .format(pretty_now()))

    if gbd_team == ['epi', 'exp']:
        valid_measures = EpiLabels.query_measures().measure_id.tolist()
        if not measure_id:
            raise ValueError("{} Epi and Exposure models require a measure_id."
                             .format(pretty_now()))
        elif measure_id not in valid_measures:
            raise ValueError("{} Measure {} not a valid measure id."
                             .format(pretty_now(), measure_id))
    elif gbd_team == 'sev':
        if not gbd_id:
            raise ValueError("{} If gbd_team is 'sev', you must provide an "
                             "rei_id as gbd_id.".format(pretty_now()))

    if age_group_id:
        if 27 in age_group_id and len(age_group_id) > 1:
            age_std_warning = ("Age-standardized found in age_group_id input. "
                               "Only generating age-standardized plot.")
            warnings.warn(age_std_warning, UserWarning)
            age_group_id = 27

    if gbd_team == 'cod' and sex_id == 3:
        raise ValueError("{} Cod models do not support sex_id 3"
                         .format(pretty_now()))

    print("{} Instantiate Metadata".format(pretty_now()))
    metadata = Metadata()

    print("{} Instantiate Model".format(pretty_now()))
    if gbd_team == 'cod':
        model = Cod(metadata,
                    model_version_id=model_version_id,
                    gbd_id=gbd_id,
                    sex_id=sex_id,
                    age_group_id=age_group_id)
        labels = CodLabels(model=model)
    elif gbd_team == 'sev':
        try:
            model = Sev(metadata,
                        gbd_id=gbd_id,
                        sex_id=sex_id)
        except Exception:
            print("{} No 2017 results returned, defaulting to gbd round 4."
                  .format(pretty_now()))
            metadata = Metadata(gbd_round_id=4)
            model = Sev(metadata,
                        gbd_id=gbd_id,
                        sex_id=sex_id)

        labels = SevLabels(model=model)
    else:
        model = Epi(metadata,
                    model_version_id=model_version_id,
                    gbd_id=gbd_id,
                    age_group_id=age_group_id,
                    measure_id=measure_id,
                    sex_id=sex_id)
        labels = EpiLabels(model=model)

    for sex in model.sex_id:
        print("{} Instantiate plot object for sex_id {}"
              .format(pretty_now(), sex))
        plot = SdiScatterPlot(model=model,
                              metadata=metadata,
                              labels=labels,
                              sex_id=sex)
        plot_and_save(plot, outdir)
    print("{} Find plot(s) in directory: {}".format(pretty_now(), outdir))


if __name__ == '__main__':
    start_time = time.time()
    create_scatter(*parse_arguments())
    end_time = time.time()
    duration = end_time - start_time
    print('{} Total runtime: {}'.format(pretty_now(), duration))
