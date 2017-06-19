using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ballScript : MonoBehaviour {
	public float ballSpeed;
	public Vector3 direction;

	// Use this for initialization
	// initializes speed and direction for each ball
	void Start () {
		ballSpeed = Random.Range (1.0f, 4.0f);
		direction = Random.insideUnitCircle.normalized;
	}
	
	// Update is called once per frame
	void Update () {
		transform.Translate (direction * ballSpeed * Time.deltaTime);
	}
	void OnBecameInvisible() {
		Destroy(gameObject);
	}
}
