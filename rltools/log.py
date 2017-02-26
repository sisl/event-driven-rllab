import os

import tableprint
import tables

from rltools import nn, util


def _printfields(fields, print_header=True):
    names, vals = [], []
    for name, val, _ in fields:
        names.append(name)
        if val is None:
            # Display None as empty
            vals.append('')
        else:
            vals.append(val)
    # Print header
    if print_header:
        print(tableprint.header(names, width=11))
    # Print value row
    print(tableprint.row(vals, width=11))


def _type_to_col(t, pos):
    if t is int:
        return tables.Int32Col(pos=pos)
    if t is float:
        return tables.Float32Col(pos=pos)
    raise NotImplementedError(t)


class TrainingLog(object):
    """A training log backed by PyTables.

    Stores diagnostics numbers as well as model snapshots
    """

    def __init__(self, filename, attrs, debug=True):
        if filename is None:
            util.warn('WARNING: not writing to any file')
            self.f = None
        else:
            if os.path.exists(filename):
                raise RuntimeError('Log file {} already exists\n'.format(filename))
            self.f = tables.open_file(filename, mode='w')
            for k, v in attrs:
                self.f.root._v_attrs[k] = v
            self.log_table = None

        self.schema = None  # list of col name / types for display
        self.debug = debug

    def close(self):
        if self.f is not None:
            self.f.close()

    def write(self, kvt, display=True, **kwargs):
        # Writing to log
        if self.f is not None:
            if self.log_table is None:
                desc = {k: _type_to_col(t, pos)
                        for pos, (k, _, t) in enumerate(kvt)}  # key, value, type
                self.log_table = self.f.create_table(self.f.root, 'log', desc)

            row = self.log_table.row
            for k, v, _ in kvt:
                row[k] = v
            row.append()

            self.log_table.flush()

        if display:
            if self.schema is None:
                self.schema = [(k, t) for k, _, t in kvt]
            else:
                # Missing columns, fill with None
                nonefilled_kvt = []
                kvt_dict = {k: (v, t) for k, v, t in kvt}
                for schema_k, schema_t in self.schema:
                    if schema_k in kvt_dict:
                        v, t = kvt_dict[schema_k]
                        nonefilled_kvt.append((schema_k, v, t))  # check t == schema_t too?
                    else:
                        nonefilled_kvt.append((schema_k, None, schema_t))
                kvt = nonefilled_kvt
            if self.debug:
                _printfields(kvt, **kwargs)

    def write_snapshot(self, sess, model, key_iter):
        if self.f is None:
            return
        if not isinstance(model, nn.Model):
            util.warn("WARNING: trying to save a non NN model. Skipping...")
            return
        # Get var values
        vs = model.get_variables()
        vals = sess.run(vs)

        # Save all variables into this group
        snapshot_root = '/snapshots/iter%07d' % key_iter

        for v, val in zip(vs, vals):
            fullpath = snapshot_root + '/' + v.name
            groupname, arrayname = fullpath.rsplit('/', 1)
            self.f.create_array(groupname, arrayname, val, createparents=True)

        # Store the model hash as an attribute
        model_root = snapshot_root + '/' + model.varscope.name
        self.f.getNode(model_root)._v_attrs.hash = model.savehash(sess)

        self.f.flush()
