import React from 'react';
import axios from "axios";

import { makeStyles } from '@material-ui/core/styles';
import ExpandMoreIcon from "@material-ui/icons/ExpandMore";
import {
  Button,
  ButtonGroup,
  Container,
  Dialog, DialogTitle,
  ExpansionPanel,
  ExpansionPanelDetails,
  ExpansionPanelSummary,
  FormControlLabel,
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
  button: {
    marginTop: theme.spacing(1),
  },
  heading: {
    fontSize: theme.typography.pxToRem(15),
    fontWeight: theme.typography.fontWeightRegular,
  },
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
    constrain_thickness: true,
    constrain_area: true,
    constrain_moment: false,
    n_proc: 28,
    report: false,
  });

  const handleChange = name => event => {
    setValues({ ...values, [name]: event.target.value });
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
      constrain_thickness: values.constrain_thickness,
      constrain_area: values.constrain_area,
      constrain_moment: values.constrain_moment,
      n_proc: values.n_proc,
      report: values.report,
    };
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
    >
      <DialogTitle>Submit New Run</DialogTitle>
      <form onSubmit={(event) => addRun(event)}>
        <ExpansionPanel>
          <ExpansionPanelSummary
            expandIcon={<ExpandMoreIcon />}
            aria-controls="panel1a-content"
            id="panel1a-header"
          >
            <Typography className={classes.heading}>Basic Problem Setup</Typography>
          </ExpansionPanelSummary>
          <ExpansionPanelDetails>
            <Container maxWidth="xs">
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
              <FormControlLabel control={
                <Switch
                  checked={values.constrain_thickness}
                  onChange={handleChange('constrain_thickness')}
                  name="constrain_thickness"
                  inputProps={{ 'aria-label': 'primary checkbox' }}
                />} label="Constrain Thickness?" //labelPlacement="top"
              />
              <br/>
              <FormControlLabel control={
                <Switch
                  checked={values.constrain_area}
                  onChange={handleChange('constrain_area')}
                  name="constrain_area"
                  inputProps={{ 'aria-label': 'primary checkbox' }}
                />} label="Constrain Area?" //labelPlacement="top"
              />
              <br/>
              <FormControlLabel control={
                <Switch
                  checked={values.constrain_moment}
                  onChange={handleChange('constrain_moment')}
                  name="constrain_moment"
                  inputProps={{ 'aria-label': 'primary checkbox' }}
                />} label="Constrain Moment?" //labelPlacement="top"
              />
            </Container>
          </ExpansionPanelDetails>
        </ExpansionPanel>

        <ExpansionPanel>
          <ExpansionPanelSummary
            expandIcon={<ExpandMoreIcon />}
            aria-controls="panel1a-content"
            id="panel1a-header"
          >
            <Typography className={classes.heading}>Number of Design Variables</Typography>
          </ExpansionPanelSummary>
          <ExpansionPanelDetails>
            <Container maxWidth="xs">
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
              />
              <br/>
              <Typography component="i">
                It is recommended to use between 3 and 6 for each.
              </Typography>
            </Container>
          </ExpansionPanelDetails>
        </ExpansionPanel>

        <ExpansionPanel>
          <ExpansionPanelSummary
            expandIcon={<ExpandMoreIcon />}
            aria-controls="panel1a-content"
            id="panel1a-header"
          >
            <Typography className={classes.heading}>Termination Settings</Typography>
          </ExpansionPanelSummary>
          <ExpansionPanelDetails>
            <Container maxWidth="xs">
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
              />
            </Container>
          </ExpansionPanelDetails>
        </ExpansionPanel>

        <ExpansionPanel>
          <ExpansionPanelSummary
            expandIcon={<ExpandMoreIcon />}
            aria-controls="panel1a-content"
            id="panel1a-header"
          >
            <Typography className={classes.heading}>Miscellaneous Settings</Typography>
          </ExpansionPanelSummary>
          <ExpansionPanelDetails>
            <Container maxWidth="xs">
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
            </Container>
          </ExpansionPanelDetails>
        </ExpansionPanel>

        <ButtonGroup fullWidth aria-label="full width outlined button group">
          <Button
            variant="contained"
            color="primary"
            onClick={handleClose}
          >Cancel</Button>
          <Button
            type="submit"
            variant="contained"
            color="primary"
          >Submit</Button>
        </ButtonGroup>
      </form>
    </Dialog>
  );
}