import os
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')
except Exception:
    pass
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import ColorConverter, ListedColormap
from matplotlib.figure import Figure
import matplotlib.patches as mpatch
from sklearn.neighbors import NearestNeighbors

from db_tools.ezfuncs import query
from db_queries import (get_cause_metadata,
                        get_covariate_estimates,
                        get_location_metadata,
                        get_model_results,
                        get_outputs,
                        get_population,
                        get_rei_metadata)


class _Model(object):
    def __init__(self,
                 metadata,
                 model_version_id=None,
                 gbd_id=None):
        self._metadata = metadata
        self._model_version_id = model_version_id
        self._gbd_id = gbd_id
        self._valid_ages = list(range(2, 21)) + [30, 31, 32, 235]
        self._index_cols = ['model_version_id', 'location_id', 'year_id',
                            'age_group_id', 'sex_id']

    @property
    def gbd_team(self):
        return self._gbd_team

    @property
    def gbd_id(self):
        return self._gbd_id

    @gbd_id.setter
    def gbd_id(self, value):
        self._gbd_id = value

    @property
    def model_version_id(self):
        return self._model_version_id

    @model_version_id.setter
    def model_version_id(self, value):
        self._model_version_id = value

    @property
    def measure_id(self):
        return self._measure_id

    @measure_id.setter
    def measure_id(self, value):
        self._measure_id = value

    @property
    def age_group_id(self):
        return self._age_group_id

    @age_group_id.setter
    def age_group_id(self, value):
        self._age_group_id = value

    @property
    def valid_ages(self):
        return self._valid_ages

    @property
    def sex_id(self):
        return self._sex_id

    @sex_id.setter
    def sex_id(self, value):
        self._sex_id = value

    @property
    def data(self):
        return self._data

    @property
    def metadata(self):
        return self._metadata

    @property
    def index_cols(self):
        return self._index_cols

    def _get_results(self):
        df = get_model_results(self.gbd_team,
                               gbd_id=self.gbd_id,
                               gbd_round_id=self.metadata.gbd_round_id,
                               model_version_id=self.model_version_id,
                               location_id=self.metadata.location_ids,
                               year_id=self.metadata.year_id)
        assert not df.empty, ("No round 5 data found for this model.")
        return df

    def _get_outputs(self):
        """For SEVs only.

        Returns 17 columns: age_group_id, location_id, measure_id, metric_id,
                            rei_id, sex_id, year_id, age_group_name, expected,
                            location_name, measure_name, rei_name, sex, val,
                            upper, lower"""
        df = get_outputs(topic='rei',
                         rei_id=self.gbd_id,
                         gbd_round_id=self.metadata.gbd_round_id,
                         measure_id=self.measure_id,
                         metric_id=3,
                         age_group_id=self.age_group_id,
                         location_id=self.metadata.location_ids,
                         year_id=self.metadata.year_id,
                         sex_id=self.sex_id)
        assert not df.empty, ("No round 5 data found for rei_id: {}"
                              .format(self.gbd_id))
        return df

    def _get_missing_ids(self):
        if not self.model_version_id:
            self.model_version_id = self.data.model_version_id.tolist()[0]
        if not self.gbd_id:
            self.gbd_id = self._get_gbd_id()
        if not self.sex_id:
            self.sex_id = self.data.sex_id.unique().tolist()
        if not self.age_group_id:
            self.age_group_id = self.data.age_group_id.unique().tolist()

    def _make_asr(self, df, mean_col):
        df = df[df.age_group_id.isin(self.valid_ages)]
        age_weights = self._get_age_weights(5)
        df = pd.merge(df, age_weights, on='age_group_id', how='left')
        df['age_group_id'] = 27
        df.reset_index(drop=True, inplace=True)
        df[mean_col] = (df[mean_col] * df['age_group_weight_value'])
        df = df.groupby(self.index_cols)[mean_col].aggregate(np.sum)
        return df.reset_index()

    def _get_age_weights(self, gbd_round_id):
        q = """
            SELECT age_group_id, age_group_weight_value
            FROM shared.age_group_weight
            WHERE gbd_round_id = {}
            """.format(gbd_round_id)
        return query(q, conn_def='cod')

    def _validate_results(self, results):
        if self.measure_id and self.gbd_team == 'epi':
            if self.measure_id not in results.measure_id.unique().tolist():
                raise ValueError("measure id {} not in model results")

        if self.age_group_id:
            if any([age not in results.age_group_id.unique()
                    for age in self.age_group_id]):
                raise ValueError(
                    "Age group {} not returned by model results {}".format(
                        self.age_group_id,
                        results.model_version_id.unique()[0]))


class Cod(_Model):

    def __init__(self,
                 metadata,
                 model_version_id=None,
                 gbd_id=None,
                 sex_id=None,
                 age_group_id=None):
        super(Cod, self).__init__(metadata=metadata,
                                  model_version_id=model_version_id,
                                  gbd_id=gbd_id)
        self._gbd_team = 'cod'
        self._age_group_id = age_group_id
        self._sex_id = [sex_id] if sex_id else None
        self._measure_id = 1
        self._data = self._get_data()
        self._get_missing_ids()

    def _get_gbd_id(self):
        q = """
            SELECT cause_id
            FROM cod.model_version
            WHERE model_version_id = {mvid}
        """.format(mvid=self.model_version_id)
        return query(q, conn_def='cod').cause_id[0]

    def _get_data(self):
        results = self._get_results()
        self._validate_results(results)

        if self.age_group_id:
            results = results[results.age_group_id.isin(self.age_group_id)]
            results = self._make_asr(results, 'mean_death_rate')
        else:
            results = self._make_asr(results, 'mean_death_rate')
        results.rename(columns={'mean_death_rate': 'plot_val'}, inplace=True)
        results['plot_val'] = results['plot_val'] * 100000
        return results[self.index_cols + ['plot_val']]


class Epi(_Model):

    def __init__(self,
                 metadata,
                 model_version_id=None,
                 gbd_id=None,
                 measure_id=None,
                 age_group_id=None,
                 sex_id=None):
        super(Epi, self).__init__(metadata=metadata,
                                  model_version_id=model_version_id,
                                  gbd_id=gbd_id)
        self._gbd_team = 'epi'
        self._measure_id = measure_id
        self._sex_id = [sex_id] if sex_id else None
        self._age_group_id = age_group_id
        self._data = self._get_data()
        self._get_missing_ids()

    def _get_gbd_id(self):
        q = """
            SELECT modelable_entity_id
            FROM epi.model_version
            WHERE model_version_id = {mvid}
        """.format(mvid=self.model_version_id)
        return query(q, conn_def='epi').modelable_entity_id[0]

    def _get_data(self):
        results = self._get_results()
        self._validate_results(results)
        results = results[results.measure_id == self.measure_id]

        if self.age_group_id:
            results = results[results.age_group_id.isin(self.age_group_id)]
            results = self._make_asr(results, 'mean')
        else:
            results = results[results.age_group_id == 27]
        results = results[results.sex_id.isin([1, 2])]
        results.rename(columns={'mean': 'plot_val'}, inplace=True)
        results['plot_val'] = results['plot_val'] * 100000
        return results[self.index_cols + ['plot_val']]


class Sev(_Model):

    def __init__(self,
                 metadata,
                 gbd_id=None,
                 sex_id=None):
        super(Sev, self).__init__(metadata=metadata,
                                  model_version_id=None,
                                  gbd_id=gbd_id)
        self._gbd_team = 'epi'
        self._risk_type = 'sev'
        self._sex_id = [sex_id] if sex_id else [1, 2, 3]
        self._measure_id = 29
        self._age_group_id = [27]
        self._index_cols = ['location_id', 'year_id', 'age_group_id', 'sex_id']
        self._data = self._get_data()

    def _get_data(self):
        outputs = self._get_outputs()
        outputs.rename(columns={'val': 'plot_val'}, inplace=True)
        return outputs[self.index_cols + ['plot_val']]


class Metadata(object):

    def __init__(self, gbd_round_id=5):
        self._gbd_round_id = gbd_round_id
        self._year_id = self._get_year_id()
        self._locations = self._get_locations()
        self._covariates = self._get_covariates()
        self._metadata = self._merge_metadata()

    @property
    def gbd_round_id(self):
        return self._gbd_round_id

    @property
    def year_id(self):
        return self._year_id

    @property
    def covariates(self):
        return self._covariates

    @property
    def locations(self):
        return self._locations

    @property
    def location_ids(self):
        return self.locations.location_id.tolist()

    @property
    def metadata(self):
        return self._metadata

    def _get_covariates(self):
        cov = get_covariate_estimates(covariate_id=881,
                                      gbd_round_id=self._gbd_round_id,
                                      location_id=self.location_ids,
                                      year_id=self.year_id)
        cov.rename(columns={'mean_value': 'sdi'}, inplace=True)
        return cov[['location_id', 'sdi']].reset_index(drop=True)

    def _get_locations(self):
        locs = get_location_metadata(location_set_id=35,
                                     gbd_round_id=self._gbd_round_id)
        locs = locs[locs.level == 3]
        locs = locs[['location_id', 'region_id', 'ihme_loc_id']]
        return locs.reset_index(drop=True)

    def _merge_metadata(self):
        return pd.merge(self.covariates, self.locations, on='location_id')

    def _get_year_id(self):
        return 2016 if self.gbd_round_id == 4 else 2017


class _Artifact(object):

    def __init__(self, gbd_id, name, name_short):
        self._name = name
        self._name_short = name_short
        self._gbd_id = gbd_id

    @property
    def gbd_id(self):
        return self._gbd_id

    @property
    def name(self):
        return self._name

    @property
    def name_short(self):
        return self._name_short


class _Labels(object):

    def __init__(self, model):
        self._model = model
        self._model_version_id = self._model.model_version_id
        self._gbd_id = self._model.gbd_id
        self._artifact = self._create_artifact()
        self._x_axis = "Socio-demographic index"

    @property
    def artifact(self):
        return self._artifact

    @property
    def measure(self):
        return self._measure

    @property
    def title(self):
        return self._get_title()

    @property
    def x_axis(self):
        return self._x_axis

    @property
    def y_axis(self):
        return self._get_ylabel()

    @property
    def filename(self):
        name_short = self.artifact.name_short
        gbd_id = self.artifact.gbd_id
        d = {}
        for sex_id in self._model.sex_id:
            d.update({sex_id: ('{name_short}_{gbd_id}_{sex}.pdf'
                               .format(name_short=name_short,
                                       gbd_id=gbd_id,
                                       sex=sex_id))})
        return d

    def _create_artifact(self):
        raise NotImplementedError

    def _get_title(self):
        artifact_name = self.artifact.name
        artifact_name = artifact_name.replace(u'\xa0', ' ')
        sex_map = {1: 'males', 2: 'females', 3: 'both sexes'}
        age = self._process_age_string()
        d = {}
        for sex_id in self._model.sex_id:
            d.update({sex_id: ("{art} {measure}\n({age})"
                               "\nin {sex} to SDI".format(
                                   art=artifact_name.title(),
                                   measure=self.measure,
                                   age=age,
                                   sex=sex_map[sex_id].title()))})
        return d

    def _process_age_string(self):
        q = """
            SELECT
                age_group_id,
                age_group_name AS name,
                age_group_alternative_name AS alt_name,
                age_group_years_start AS year_start,
                age_group_years_end AS year_end
            FROM
                shared.age_group
        """
        ages = query(q, conn_def='cod')
        ages = ages[ages.age_group_id.isin(self._model.valid_ages)]
        age_dict = ages.set_index('age_group_id').to_dict()
        if 27 in self._model.age_group_id or not self._model.age_group_id:
            return 'Age-standardized'
        else:
            age_str = 'Age-std aggregate '
            min_age = min(self._model.age_group_id)
            max_age = max(self._model.age_group_id)
            if max_age < 5:
                age_str += '{} to {}'.format(age_dict['name'][min_age],
                                             age_dict['name'][max_age])
            elif min_age < 5:
                age_str += '{} to {} years'.format(
                    age_dict['name'][min_age],
                    int(age_dict['year_end'][max_age]))
            else:
                age_str += '{} to {} years'.format(
                    int(age_dict['year_start'][min_age]),
                    int(age_dict['year_end'][max_age]))
            return age_str


class CodLabels(_Labels):

    def __init__(self, model):
        super(CodLabels, self).__init__(model)

    def _create_artifact(self):
        q = """
            SELECT
                mv.cause_id AS gbd_id,
                s.cause_name AS name,
                s.acause AS name_short
            FROM cod.model_version mv
            JOIN shared.cause s USING (cause_id)
            WHERE model_version_id = {mvid}
        """.format(mvid=self._model_version_id)
        results = query(q, conn_def='cod')
        return _Artifact(results.gbd_id[0],
                         results.name[0],
                         results.name_short[0])

    @property
    def measure(self):
        return 'Deaths'

    def _get_ylabel(self):
        return "Deaths (ASDR * 100,000)"


class EpiLabels(_Labels):

    def __init__(self, model):
        super(EpiLabels, self).__init__(model)
        self._valid_measures = self.query_measures()
        self._measure = self._get_measure()

    @property
    def valid_measures(self):
        return self._valid_measures

    @classmethod
    def query_measures(cls):
        q = """
            SELECT
                measure_id,
                measure_name,
                measure_name_short
            FROM shared.measure
        """
        return query(q, conn_def='epi')

    def _get_measure(self):
        measures = self.valid_measures
        measures = measures[measures.measure_id == self._model.measure_id]
        return measures.measure_name.unique()[0]

    def _create_artifact(self):
        results = self._query_me_cause()
        if len(results) == 0:
            results = self._query_me_rei()
        return _Artifact(results.gbd_id[0],
                         results.name[0],
                         results.name_short[0])

    def _query_me_cause(self):
        q = """
            SELECT
                mv.modelable_entity_id AS gbd_id,
                me.modelable_entity_name AS name,
                s.acause AS name_short
            FROM epi.model_version mv
            JOIN epi.modelable_entity me USING (modelable_entity_id)
            JOIN epi.modelable_entity_cause mec USING (modelable_entity_id)
            JOIN shared.cause s USING (cause_id)
            WHERE model_version_id = {mvid}
        """.format(mvid=self._model_version_id)
        return query(q, conn_def='epi')

    def _query_me_rei(self):
        q = """
            SELECT
                mv.modelable_entity_id AS gbd_id,
                me.modelable_entity_name AS name,
                s.rei AS name_short
            FROM epi.model_version mv
            JOIN epi.modelable_entity me USING (modelable_entity_id)
            JOIN epi.modelable_entity_rei mer USING (modelable_entity_id)
            JOIN shared.rei s USING (rei_id)
            WHERE model_version_id = {mvid}
        """.format(mvid=self._model_version_id)
        return query(q, conn_def='epi')

    def _get_ylabel(self):
        return self.measure + " (rate per 100,000)"


class ExposureLabels(EpiLabels):

    def __init__(self, model):
        super(ExposureLabels, self).__init__(model)

    def _create_artifact(self):
        if self._model_version_id:
            results = self._query_rei_from_mvid()
        else:
            results = self._query_rei_from_me()
        return Artifact(results.gbd_id[0],
                        results.name[0],
                        results.name_short[0])

    def _query_rei_from_me(self):
        q = """
            SELECT
                rei_id,
                rei_name,
                rei
            FROM shared.rei
            JOIN epi.modelable_entity_rei USING (rei_id)
            JOIN epi.modelable_entity USING (modelable_entity_id)
            WHERE modelable_entity_id = {me_id}
        """.format(me_id=self._gbd_id)
        results = query(q, conn_def='epi')
        return results

    def _query_rei_from_mvid(self):
        q = """
            SELECT
                rei_id AS gbd_id,
                rei_name AS name,
                rei AS name_short
            FROM shared.rei
            JOIN epi.modelable_entity_rei USING (rei_id)
            JOIN epi.model_version USING (modelable_entity_id)
            WHERE model_version_id = {mvid}
        """.format(mvid=self._modelable_entity_id)
        results = query(q, conn_def='epi')
        return results


class SevLabels(_Labels):

    def __init__(self, model):
        super(SevLabels, self).__init__(model)
        self._measure = 'SEV'

    def _create_artifact(self):
        results = self._query_rei()
        return _Artifact(results.gbd_id[0],
                         results.name[0],
                         results.name_short[0])

    def _query_rei(self):
        q = """
            SELECT
                s.rei_id AS gbd_id,
                s.rei_name AS name,
                s.rei AS name_short
            FROM shared.rei s
            WHERE rei_id = {rei_id}
        """.format(rei_id=self._model.gbd_id)
        return query(q, conn_def='epi')

    def _get_ylabel(self):
        return 'SEV (per capita)'

    def _get_title(self):
        artifact_name = self.artifact.name
        sex_map = {1: 'males', 2: 'females', 3: 'both sexes'}
        age = self._process_age_string()
        d = {}
        for sex_id in self._model.sex_id:
            d.update({sex_id: ("{art} {measure}\n({age})"
                               "\nin {sex} to SDI".format(
                                   art=artifact_name.title(),
                                   measure=self.measure,
                                   age=age,
                                   sex=sex_map[sex_id].title()))})
        return d


class SdiScatterPlot(object):

    REGIONS = ['East Asia', 'Southeast Asia', 'Oceania', 'Central Asia',
               'Central Europe', 'Eastern Europe', 'High-income Asia Pacific',
               'Australasia', 'Western Europe', 'Southern Latin America',
               'High-income North America', 'Caribbean',
               'Andean Latin America', 'Central Latin America',
               'Tropical Latin America', 'North Africa and Middle East',
               'South Asia', 'Central Sub-Saharan Africa',
               'Eastern Sub-Saharan Africa', 'Southern Sub-Saharan Africa',
               'Western Sub-Saharan Africa']

    COLORS = ['#9E0142', '#B31C42', '#C93742', '#DE5242', '#F46D43', '#F68955',
              '#F9A667', '#FBC379', '#FEE08B', '#F8E58E', '#F2EA91', '#ECEF94',
              '#E6F598', '#C6E89B', '#A6DB9E', '#86CEA1', '#66C2A5', '#64A5A4',
              '#6288A3', '#606BA2', '#5E4FA2']

    def __init__(self, model, metadata, labels, sex_id):
        """Given Model, Metadata, and Label object, generates the plot."""
        self._model = model
        self._metadata = metadata.metadata
        self._sex_id = sex_id if sex_id else self._model.sex_id
        self._data = self._preprocess_data()
        self._labels = labels
        self._fig = Figure(figsize=(6.4, 5.6))
        FigureCanvas(self.fig)
        self._ax = self.fig.add_subplot(111)
        self._cmap = self._create_cmap()

    @property
    def data(self):
        return self._data

    @property
    def sex_id(self):
        return self._sex_id

    @property
    def labels(self):
        return self._labels

    @property
    def fig(self):
        return self._fig

    @property
    def ax(self):
        return self._ax

    @property
    def cmap(self):
        return self._cmap

    def _preprocess_data(self):
        df = pd.merge(self._model.data,
                      self._metadata,
                      on='location_id',
                      how='left')
        return df[df.sex_id == self.sex_id].reset_index(drop=True)

    def _create_cmap(self):
        cc = ColorConverter()
        return ListedColormap([cc.to_rgb(c) for c in self.COLORS],
                              'region_color_code', N=21)

    def _create_legend_patches(self):
        return [mpatch.Patch(color=self.COLORS[i], label=self.REGIONS[i])
                for i in range(len(self.COLORS))]

    def _add_normalize_y(self):
        min_val = min(self.data.plot_val)
        max_val = max(self.data.plot_val)
        self.data['normalized_y'] = self.data.plot_val.apply(
            lambda x: (x - min_val) / (max_val - min_val))

    def _label_lonely_locations(self):
        self._add_normalize_y()
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(
            self.data[['sdi', 'normalized_y']])
        r_nbrs = nbrs.radius_neighbors(self.data[['sdi', 'normalized_y']],
                                       return_distance=False,
                                       radius=0.03)
        lone_samples = []
        for i in range(len(r_nbrs)):
            if len(r_nbrs[i]) == 1:
                lone_samples.append(r_nbrs[i][0])

        for i in lone_samples:
            self.ax.annotate(self.data.loc[i, 'ihme_loc_id'],
                             alpha=0.7,
                             fontsize=7,
                             xy=self.data.loc[i, ['sdi', 'plot_val']],
                             xytext=(self.data.loc[i, ['sdi']] - 0.0175,
                                     (self.data.loc[i, ['plot_val']] -
                                     (max(self.data.plot_val) * 0.04))))

    def graph(self):
        self.ax.scatter(self.data.sdi,
                        self.data.plot_val,
                        s=50,
                        c=self.data.region_id,
                        cmap=self.cmap,
                        alpha=0.4,
                        linewidths=0)

        max_val, min_val = max(self.data.plot_val), min(self.data.plot_val)

        if min_val == 0:
            min_limit = max_val * .965 - max_val
        else:
            min_limit = min_val + min_val * .035
        self.ax.set_ylim(min_limit, max_val + max_val * .035)
        self.ax.set_xlim(0, 1)

        self.label()

    def label(self):
        # create nearest neighbors labeling
        self._label_lonely_locations()

        # label axes and create title
        self.ax.set_title(self.labels.title[self.sex_id], fontsize=12)
        self.ax.set_xlabel(self.labels.x_axis, fontsize=12)
        self.ax.set_ylabel(self.labels.y_axis, fontsize=12)

    def save(self, outdir):
        outpath = os.path.join(outdir, self.labels.filename[self.sex_id])
        self.fig.savefig(outpath, format='pdf', dpi=300)
