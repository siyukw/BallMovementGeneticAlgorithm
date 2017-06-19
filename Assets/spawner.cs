using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class spawner : MonoBehaviour {
	public GameObject ball;
	public GameObject fakeBall;
	public float waitingForNextSpawn = 1;
	public float Countdown = 1;
	public float xMin;
	public float xMax;
	public float yMin;
	public float yMax;

	public void Start () {
		// Start calling the Spawn function repeatedly after a delay .
		// InvokeRepeating("Spawn", Time.deltaTime * 0.2);
	}
	public void Update() {
		// timer to spawn the next goodie Object
		Countdown -= Time.deltaTime;
		if(Countdown <= 0) {
			Spawn();
			Countdown = waitingForNextSpawn;
		}
	}
	void Spawn () {
		// Instantiate a random enemy.
		Vector3 pos = new Vector3 (Random.Range (xMin, xMax), Random.Range (yMin, yMax), 0.0f);
		fakeBall = Instantiate (ball, pos, Quaternion.identity);
	}

}