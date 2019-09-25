import React from 'react';
import axios from "axios";

import { makeStyles } from '@material-ui/core/styles';
import {
  Button,
  Dialog, DialogActions, DialogContent, DialogTitle,
  FormControlLabel, Paper,
  Switch,
  TextField,
  Typography
} from "@material-ui/core";

const useStyles = makeStyles(theme => ({
  dialog: {
    padding: theme.spacing(2),
    textAlign: 'center',
    color: theme.palette.text.secondary,
  },
  textField: {
    marginLeft: theme.spacing(1),
    marginRight: theme.spacing(1),
  },
  buttonGroup: {
    margin: theme.spacing(1),
  },
  heading: {
    fontSize: theme.typography.pxToRem(15),
    fontWeight: theme.typography.fontWeightRegular,
  },
  paper: {
    padding: theme.spacing(2),
    margin: theme.spacing(2),
  }
}));

export default function AddRun(props) {
  const classes = useStyles();
  const { onClose, open } = props;
  const [values, setValues] = React.useState({
    cl: 1.0,
    n_c: 3,
    n_t: 3,
    fix_te: true,
    gen: 1000,
    tolx: 1e-8,
    tolf: 1e-8,
    t_te_min: 0.0025,
    t_c_min: 0.01,
    A_cs_min: null,
    Cm_max: null,
    n_proc: 28,
    report: false,
  });
  const [TELabel, setTELabel] = React.useState("TE Thickness");

  const handleChange = name => event => {
    if (['fix_te', 'report'].includes(name)) {
      setValues({ ...values, [name]: !values[name]});

      if (name === 'fix_te') {
        if (values[name]) {
          setTELabel("Minimum TE Thickness");
        } else {
          setTELabel("TE Thickness");
        }
      }
    } else {
      setValues({...values, [name]: event.target.value});
    }
  };

  const handleClose = () => {
    onClose();
  };

  const addRun = event => {
    event.preventDefault();
    const data = {
      cl: values.cl,
      n_c: values.n_c,
      n_t: values.n_t,
      fix_te: values.fix_te,
      gen: values.gen,
      tolx: values.tolx,
      tolf: values.tolf,
      t_te_min: values.t_te_min ? Number(values.t_te_min) : null,
      t_c_min: values.t_c_min ? Number(values.t_c_min) : null,
      A_cs_min: values.A_cs_min ? Number(values.A_cs_min) : null,
      Cm_max: values.Cm_max ? Number(values.Cm_max) : null,
      n_proc: values.n_proc,
      report: values.report,
    };
    console.log(data);
    axios.post(`${process.env.REACT_APP_RUNS_SERVICE_URL}/runs`, data)
      .then((res) => {

      })
      .catch((err) => { console.log(err); });
    onClose();
  };

  return (
    <Dialog
      className={classes.dialog}
      open={open}
      onClose={handleClose}
    >
      <DialogTitle>Submit New Run</DialogTitle>
      <form onSubmit={(event) => addRun(event)}>
        <DialogContent>
          <Paper
            container
            maxWidth="xs"
            className={classes.paper}
          >
            <Typography variant="h7">Basic Problem Setup</Typography>
            <br/>
            <TextField
              label="Design Lift Coefficient"
              name="cl"
              value={values.cl}
              onChange={handleChange('cl')}
              type="number"
              inputProps={{
                step: 0.001,
              }}
              className={classes.textField}
              InputLabelProps={{
                shrink: true,
              }}
              margin="normal"
              variant="outlined"
            />
            <br/>
            <FormControlLabel control={
              <Switch
                checked={values.fix_te}
                onChange={handleChange('fix_te')}
                value="fix_te"
                name="fix_te"
                inputProps={{ 'aria-label': 'primary checkbox' }}
              />} label="Fix TE Thickness?" //labelPlacement="top"
            />
            <br/>
            <TextField
              label={TELabel}
              name="t_te_min"
              value={values.t_te_min}
              onChange={handleChange('t_te_min')}
              type="text"
              inputProps={{
                pattern: "-?([1-9]\\d*|0)+(\\.\\d*)?([Ee][+-]?\\d+)?",
              }}
              className={classes.textField}
              InputLabelProps={{
                shrink: true,
              }}
              margin="normal"
              variant="outlined"
            />
            <br/>
            <TextField
              label="Minimum t/c"
              name="t_c_min"
              value={values.t_c_min}
              onChange={handleChange('t_c_min')}
              type="text"
              inputProps={{
                pattern: "[+-]?([1-9]\\d*|0)?(\\.\\d*)?([Ee][+-]?\\d+)?",
              }}
              className={classes.textField}
              InputLabelProps={{
                shrink: true,
              }}
              margin="normal"
              variant="outlined"
            />
            <br/>
            <TextField
              label="Minimum Area"
              name="A_cs_min"
              value={values.A_cs_min}
              onChange={handleChange('A_cs_min')}
              type="text"
              inputProps={{
                pattern: "[+-]?([1-9]\\d*|0)?(\\.\\d*)?([Ee][+-]?\\d+)?",
              }}
              className={classes.textField}
              InputLabelProps={{
                shrink: true,
              }}
              margin="normal"
              variant="outlined"
            />
            <br/>
            <TextField
              label="Maximum Absolute Cm"
              name="Cm_max"
              value={values.Cm_max}
              onChange={handleChange('Cm_max')}
              type="text"
              inputProps={{
                pattern: "[+-]?([1-9]\\d*|0)?(\\.\\d*)?([Ee][+-]?\\d+)?",
              }}
              className={classes.textField}
              InputLabelProps={{
                shrink: true,
              }}
              margin="normal"
              variant="outlined"
            />
          </Paper>
          <Paper
            container
            maxWidth="xs"
            className={classes.paper}
          >
            <Typography variant="h7">Number of Design Variables</Typography>
            <br/>
            <TextField
              label="Mean Chord Line"
              name="n_c"
              value={values.n_c}
              onChange={handleChange('n_c')}
              type="number"
              inputProps={{
                min: 1,
              }}
              className={classes.textField}
              InputLabelProps={{
                shrink: true,
              }}
              margin="normal"
              variant="outlined"
            />
            <br/>
            <TextField
              label="Thickness Distribution"
              name="n_t"
              value={values.n_t}
              onChange={handleChange('n_t')}
              type="number"
              inputProps={{
                min: 1,
              }}
              className={classes.textField}
              InputLabelProps={{
                shrink: true,
              }}
              margin="normal"
              variant="outlined"
            />
          </Paper>
          <Paper
            container
            maxWidth="xs"
            className={classes.paper}
          >
            <Typography variant="h7">Termination Settings</Typography>
            <br/>
            <TextField
              label="Number of Generations"
              name="gen"
              value={values.gen}
              onChange={handleChange('gen')}
              type="number"
              inputProps={{
                min: 0,
              }}
              className={classes.textField}
              InputLabelProps={{
                shrink: true,
              }}
              margin="normal"
              variant="outlined"
            />
            <br/>
            <TextField
              label="Design Vector Tolerance"
              name="tolx"
              value={values.tolx}
              onChange={handleChange('tolx')}
              type="text"
              inputProps={{
                pattern: "[+-]?([1-9]\\d*|0)?(\\.\\d*)?([Ee][+-]?\\d+)?",
              }}
              className={classes.textField}
              InputLabelProps={{
                shrink: true,
              }}
              margin="normal"
              variant="outlined"
            />
            <br/>
            <TextField
              label="Objective Function Tolerance"
              name="tolf"
              value={values.tolf}
              onChange={handleChange('tolf')}
              type="text"
              inputProps={{
                pattern: "[+-]?([1-9]\\d*|0)?(\\.\\d*)?([Ee][+-]?\\d+)?",
              }}
              className={classes.textField}
              InputLabelProps={{
                shrink: true,
              }}
              margin="normal"
              variant="outlined"
            />
          </Paper>
          <Paper
            container
            maxWidth="xs"
            className={classes.paper}
          >
            <Typography variant="h7">Miscellaneous</Typography>
            <TextField
              label="Number of Processors"
              name="n_proc"
              value={values.n_proc}
              onChange={handleChange('n_proc')}
              type="number"
              inputProps={{
                min: 1,
              }}
              className={classes.textField}
              InputLabelProps={{
                shrink: true,
              }}
              margin="normal"
              variant="outlined"
            />
            <br/>
            <FormControlLabel control={
              <Switch
                checked={values.report}
                onChange={handleChange('report')}
                value="report"
                name="report"
                inputProps={{ 'aria-label': 'primary checkbox' }}
              />} label="Report Results via Email?" //labelPlacement="top"
            />
          </Paper>
        </DialogContent>
        <DialogActions>
          <Button
            color="primary"
            onClick={handleClose}
          >Cancel</Button>
          <Button
            type="submit"
            color="primary"
          >Submit</Button>
        </DialogActions>
      </form>
    </Dialog>
  );
}