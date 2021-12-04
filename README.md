<h3 align="center">CS245-Geolocation-Predictor</h3>
  <p align="center">
    Predict User Geolocation by vanilla and Spatial label propagation
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#credits">Credits</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About The Project


Motivated by a recent line of work on explaining/interpreting neural networks, we decide to investigate the problem of explaining a geolocation tagger. We will have three key milestones in this project: first, we propose Explained Label Propagation algorithm, which combines vanilla label propagation with some book-keeping technique to explain each decision it makes. More specifically, the algorithm not only predicts the class of a new user but also keep track of the contribution (percentage) of each labeled user for the algorithm to come to this decision. Second, we plan to apply our Explained Label Propagation to spatial label propagation proposed by Jurgens@2013 and experimented with different heuristics for select one's location from one's neighbors. Finally, we plan to evaluate our our Explained Label Propagation on GEOTEXT datasets.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Credits

* Haiying Huang, UCLA
* Yu Hou, UCLA
* Cheng Lu, UCLA
* Zuer Wang, UCLA
* Yuhan Shao, UCLA

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

If you have discovered a bug in the build software or want to report an error in
the library, please create a new
[Issue](https://github.com/arthurhaiying/CS245-Geolocation-Predictor/issues) on our github page.

<p align="right">(<a href="#top">back to top</a>)</p>