import React from 'react';
import {
  Button,
  Card,
  CardActions,
  CardActionArea,
  CardContent,
  CardMedia,
  makeStyles,
  Paper,
  Typography,
} from "@material-ui/core";

const useStyles = makeStyles(theme => ({
  card: {
    maxWidth: 345,
    margin: theme.spacing(2),
  },
  media: {
    height: 50,
  },
}));

export default function Run(props) {
  const classes = useStyles();
  const { run } = props;

  return (
    <Card className={classes.card}>
      <CardActionArea>
        <CardMedia
          className={classes.media}
          // image="/static/images/cards/contemplative-reptile.jpg"
          title="Optimization Run"
          style={
            {
              backgroundColor: run.status === 0 ? 'lightblue' : (
                run.status === 1 ? 'orange' : (
                  run.status === 2 ? 'lightgreen' : 'red'
                )
              )
            }
          }
        >
          <Paper />
        </CardMedia>
        <CardContent>
          <Typography gutterBottom variant="h5" component="h2">
            Optimization Run
          </Typography>
          <Typography variant="body2" color="textSecondary" component="p">
            Run status:
            {run.status === 0 && " In Queue"}
            {run.status === 1 && " Running..."}
            {run.status === 2 && " Completed..."}
            {![0, 1, 2].includes(run.status) && " Failed"}
            <br/>
          </Typography>
        </CardContent>
      </CardActionArea>
      <CardActions>
        <Button size="small" color="primary">
          Details
        </Button>
      </CardActions>
    </Card>
  );
}