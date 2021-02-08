import React, { useState } from "react";

import Avatar from "@material-ui/core/Avatar";
import Button from "@material-ui/core/Button";
import TextField from "@material-ui/core/TextField";
import Grid from "@material-ui/core/Grid";
import Typography from "@material-ui/core/Typography";
import { makeStyles } from "@material-ui/core/styles";
import CircularProgress from "@material-ui/core/CircularProgress";
import TwitterIcon from "@material-ui/icons/Twitter";
import Alert from "@material-ui/lab/Alert";
import FormLabel from "@material-ui/core/FormLabel";
import FormControl from "@material-ui/core/FormControl";
import FormGroup from "@material-ui/core/FormGroup";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import FormHelperText from "@material-ui/core/FormHelperText";
import Checkbox from "@material-ui/core/Checkbox";
import Paper from "@material-ui/core/Paper";

import axios from "axios";

const useStyles = makeStyles((theme) => ({
  paper: {
    marginTop: theme.spacing(8),
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
  },
  avatar: {
    alignItems: "center",
    margin: theme.spacing(1),
    backgroundColor: theme.palette.secondary.main,
  },
  form: {
    width: "100%", // Fix IE 11 issue.
    marginTop: theme.spacing(1),
  },
  submit: {
    margin: theme.spacing(3, 0, 2),
  },
  formControl: {
    margin: theme.spacing(3),
  },
  alert: {
    width: "100%",
    margin: theme.spacing(1),
  },
}));

const Prediction = () => {
  const classes = useStyles();

  const url = process.env.REACT_APP_BACKEND_URL + "/prediction";

  const handleChange = (event) => {
    setModels({ ...models, [event.target.name]: event.target.checked });
  };

  const [tweet, setTweet] = useState("");
  const [prediction, setPrediction] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState([]);
  const [models, setModels] = useState({
    decisiontreeclassifier_using_cbow_vector: false,
    decisiontreeclassifier_using_count_vector: false,
    decisiontreeclassifier_using_skip_gram_vector: false,
    decisiontreeclassifier_using_tfidf_vector: false,
    kneighborsclassifier_using_cbow_vector: false,
    kneighborsclassifier_using_count_vector: false,
    kneighborsclassifier_using_skip_gram_vector: false,
    kneighborsclassifier_using_tfidf_vector: false,
    logisticregression_using_cbow_vector: false,
    logisticregression_using_count_vector: false,
    logisticregression_using_skip_gram_vector: false,
    logisticregression_using_tfidf_vector: false,
    multinomialnb_using_count_vector: false,
    multinomialnb_using_tfidf_vector: false,
    randomforestclassifier_using_cbow_vector: false,
    randomforestclassifier_using_count_vector: false,
    randomforestclassifier_using_skip_gram_vector: false,
    randomforestclassifier_using_tfidf_vector: false,
    svc_using_cbow_vector: false,
    svc_using_count_vector: false,
    svc_using_skip_gram_vector: false,
    svc_using_tfidf_vector: false,
    xgbclassifier_using_cbow_vector: false,
    xgbclassifier_using_count_vector: false,
    xgbclassifier_using_skip_gram_vector: false,
    xgbclassifier_using_tfidf_vector: false,
    // lstm: false,
  });
  const {
    decisiontreeclassifier_using_cbow_vector,
    decisiontreeclassifier_using_count_vector,
    decisiontreeclassifier_using_skip_gram_vector,
    decisiontreeclassifier_using_tfidf_vector,
    kneighborsclassifier_using_cbow_vector,
    kneighborsclassifier_using_count_vector,
    kneighborsclassifier_using_skip_gram_vector,
    kneighborsclassifier_using_tfidf_vector,
    logisticregression_using_cbow_vector,
    logisticregression_using_count_vector,
    logisticregression_using_skip_gram_vector,
    logisticregression_using_tfidf_vector,
    multinomialnb_using_count_vector,
    multinomialnb_using_tfidf_vector,
    randomforestclassifier_using_cbow_vector,
    randomforestclassifier_using_count_vector,
    randomforestclassifier_using_skip_gram_vector,
    randomforestclassifier_using_tfidf_vector,
    svc_using_cbow_vector,
    svc_using_count_vector,
    svc_using_skip_gram_vector,
    svc_using_tfidf_vector,
    xgbclassifier_using_cbow_vector,
    xgbclassifier_using_count_vector,
    xgbclassifier_using_skip_gram_vector,
    xgbclassifier_using_tfidf_vector,
    // lstm,
  } = models;

  const handelSubmit = (e) => {
    e.preventDefault();
    setPrediction([]);
    predict();
  };
  const predict = async () => {
    setIsLoading(true);

    try {
      let response = await axios.get(url, {
        params: {
          tweet: tweet,
          models: Object.keys(models)
            .filter((key) => models[key])
            .join(","),
        },
      });

      if (response.data && response.data.prediction)
        setPrediction(response.data.prediction);
      if (response.data && response.data.message)
        setErrors([...errors, response.data.message]);
      if (response.data && response.data.exception)
        setErrors([...errors, response.data.exception]);
    } catch (error) {
      console.log(error);
      setErrors([...errors, error.message]);
    } finally {
      setIsLoading(false);
      if (errors.length > 0)
        setTimeout(() => {
          setErrors([]);
        }, 3000);
    }
  };

  const keyToName = (key) =>
    key
      .replaceAll("_", " ")
      .replaceAll("classifier", " classifier")
      .replaceAll("multinomial", "multinomial ")
      .replaceAll("regression", " regression")
      .replaceAll(" nb ", " NB ");
  return (
    <div className={classes.paper}>
      <Grid container spacing={3}>
        <Grid item xs={12} className={classes.paper}>
          <Avatar className={classes.avatar}>
            <TwitterIcon />
          </Avatar>
          <Typography component="h1" variant="h5">
            Disaster Tweets
          </Typography>
        </Grid>
        <Grid item xs={6}>
          {isLoading && (
            <Grid style={{ display: "flex", justifyContent: "center" }}>
              <CircularProgress />
            </Grid>
          )}
          {errors.map((error, index) => (
            <Alert className={classes.alert} key={index} severity="error">
              {error}
            </Alert>
          ))}

          <form className={classes.form} noValidate>
            <TextField
              disabled={isLoading}
              variant="outlined"
              margin="normal"
              multiline
              rows={4}
              required
              fullWidth
              id="tweet"
              label="Tweet"
              name="tweet"
              autoFocus
              onChange={(e) => setTweet(e.target.value)}
            />

            <Button
              type="submit"
              fullWidth
              variant="contained"
              color="primary"
              className={classes.submit}
              onClick={handelSubmit}
            >
              Predict
            </Button>
            {prediction &&
              prediction.map((p) => {
                const key = Object.keys(p)[0];
                const model_name = keyToName(key);
                const isDisasterTweet = p[key];
                return (
                  <Alert
                    className={classes.alert}
                    severity={isDisasterTweet ? "warning" : "info"}
                  >
                    {model_name} result :
                    {isDisasterTweet
                      ? " A real disaster Tweet"
                      : " Not a real disaster Tweet"}
                  </Alert>
                );
              })}
          </form>
        </Grid>
        <Grid item xs={6}>
          <FormControl component="fieldset" className={classes.formControl}>
            <FormLabel component="legend">Models</FormLabel>
            <FormGroup>
              {Object.keys(models).map((key) => (
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={models[key]}
                      onChange={handleChange}
                      name={key}
                    />
                  }
                  label={keyToName(key)}
                />
              ))}
            </FormGroup>
          </FormControl>
        </Grid>
      </Grid>
    </div>
  );
};

export default Prediction;
